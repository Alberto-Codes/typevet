"""Live web demo: typevet typed judgments on local Gemma 4, text and image.

The page asks one typed Choice about an expense claim and a receipt image, and
three typed questions (Noul, Choice and Score) about a customer message. A
"behind the scenes" panel shows the exact prompt, the candidate tokens, their
raw log probabilities and the router calls (see ``bts.py``).

Run the command from the repository root, because the gallery reads the CORD
receipts from ``tests/fixtures/cord/expense_smoke/``:

    uv run python examples/live-demo/server.py

The server binds 127.0.0.1 only. ``LIVE_UI_PORT`` sets the port (default
8765). Model calls go to the llama.cpp router that ``TYPEVET_LLAMA__BASE_URL``
names (default ``http://127.0.0.1:8090``). ``TYPEVET_LLAMA__MULTIMODAL_MODEL``
names the model (default ``gemma-4-31b-kv9-q4km-mm``). Each judgment writes a
JSON receipt under ``typevet-receipts/``.

Examples:
    ```bash
    LIVE_UI_PORT=8766 uv run python examples/live-demo/server.py
    ```

See Also:
    - [typevet.runtime.open_gemma_native_vision_judgment][]: Judgment session.
    - examples/live-demo/README.md: Prerequisites and run of show.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import bts
import httpx
from PIL import Image, ImageOps

from typevet.adapters.inbound import load_llama_settings
from typevet.domain import (
    Choice,
    GenerationError,
    ImageInput,
    JudgmentResponse,
    JudgmentValidationError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoringValidationError,
)
from typevet.runtime import GemmaNativeVisionSession, open_gemma_native_vision_judgment

DEFAULT_ROUTER = "http://127.0.0.1:8090"
DEFAULT_MODEL = "gemma-4-31b-kv9-q4km-mm"
DEFAULT_TIMEOUT_S = "900"
HOST = "127.0.0.1"
PORT = int(os.environ.get("LIVE_UI_PORT", "8765"))
HERE = Path(__file__).resolve().parent
REPO = Path.cwd()
RECEIPT_DIR = REPO / "typevet-receipts"
HALF = 0.5
THUMB_WIDTH = 360
CORD = REPO / "tests" / "fixtures" / "cord" / "expense_smoke"
MAX_BODY = 25 * 1024 * 1024
RECEIPT_IDS = [f"R0{i}" for i in range(1, 7)]

VERDICT_CRITERIA = {
    "supported": "The receipt image shows this total",
    "contradicted": "The receipt image shows a different total",
    "insufficient_evidence": "The image does not show enough to decide",
}
VERDICT_INSTRUCTIONS = "Look at the receipt image. Does it support the expense claim?"

FRAUD_CRITERIA = {
    "unauthorized_transaction": "A charge the customer did not make or approve",
    "duplicate_charge": "The same purchase was billed more than once",
    "phishing_or_scam": "Customer was tricked into paying or sharing data",
    "account_takeover": "Someone else gained control of the account",
    "not_fraud": "No fraud or billing error is described",
    "unclear": "Not enough information to decide",
}
URGENCY_LEVELS = [
    "none: no money at risk",
    "low: small issue, no money lost",
    "high: money already lost",
    "critical: ongoing loss, act now",
]
TEXT_QUESTIONS: dict[str, Question] = {
    "unauthorized": Noul(
        instructions="Does the customer report a transaction they did not authorize?",
        criteria={
            "true": "Yes, they report a charge they did not authorize",
            "false": "No unauthorized transaction is reported",
        },
    ),
    "fraud_type": Choice(
        instructions="Which type of issue does the customer report?",
        criteria=FRAUD_CRITERIA,
    ),
    "urgency": Score(
        instructions="How urgent is this customer's issue?",
        criteria=URGENCY_LEVELS,
    ),
}
MAGIC = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),
}


class DemoError(Exception):
    """One-line error shown to the user.

    Examples:
        ```python
        raise DemoError("enter a claim to check")
        ```
    """


def gpu_name() -> str:
    """Return the first GPU name from ``nvidia-smi``, or ``unknown``.

    Returns:
        The GPU name, or ``unknown`` when the tool is missing.
    """
    try:
        out = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


class Demo:
    """Holds one warm judgment session and serializes calls with a lock.

    Attributes:
        lock (threading.Lock): Serializes judgments, so captures do not mix.
        http_log (list[str]): One line for each router request.
        stack (ExitStack): Owns the HTTP client and the session.
        settings (LlamaSettings): Router settings from the environment.
        client (httpx.Client): Router client with the capture hooks.
        capture (bts.Capture): Behind-the-scenes capture.
        build (str): llama.cpp build from ``/props``.
        gpu (str): GPU name, or ``unknown``.
        manifest (dict[str, Any]): CORD receipt manifest.
        receipts (dict[str, Any]): Manifest receipts by id.
        session (GemmaNativeVisionSession): Warm judgment session.
        warmup_s (float): Seconds to open the session and judge once.

    Examples:
        ```python
        demo = Demo()
        print(demo.status()["model"])
        ```
    """

    def __init__(self) -> None:
        """Open the router client and the session, then judge once to warm up."""
        self.lock = threading.Lock()
        self.http_log: list[str] = []
        self.stack = ExitStack()
        env = dict(os.environ)
        env.setdefault("TYPEVET_LLAMA__BASE_URL", DEFAULT_ROUTER)
        env.setdefault("TYPEVET_LLAMA__MULTIMODAL_MODEL", DEFAULT_MODEL)
        env.setdefault("TYPEVET_LLAMA__TIMEOUT", DEFAULT_TIMEOUT_S)
        self.settings = load_llama_settings(env)
        router = self.settings.base_url
        model = self.settings.multimodal_model
        self.client = self.stack.enter_context(
            httpx.Client(base_url=router, timeout=self.settings.timeout)
        )
        self.client.event_hooks["request"].append(
            lambda r: self.http_log.append(f"{r.method} {r.url.path}")
        )
        self.capture = bts.Capture()
        self.capture.install(self.client)
        props = self.client.get("/props").raise_for_status().json()
        self.build = str(props.get("build_info", "unknown"))
        self.gpu = gpu_name()
        self.manifest = json.loads((CORD / "manifest.json").read_text())
        self.receipts = {r["receipt_id"]: r for r in self.manifest["receipts"]}
        print(
            f"Opening session on {router} model {model} (may take minutes)...",
            flush=True,
        )
        t0 = time.perf_counter()
        self.session: GemmaNativeVisionSession = self.stack.enter_context(
            open_gemma_native_vision_judgment(
                settings=self.settings,
                model=model,
                http_client=self.client,
                scoring_port_wrapper=self.capture.wrap,
            )
        )
        warm_q = {
            "greeting": Noul(
                instructions="Is this text a greeting?",
                criteria={"true": "It is a greeting", "false": "It is not a greeting"},
            )
        }
        warm = self.session.port.judge("Hello there!", warm_q, self.session.model)
        ans = warm.answers["greeting"]
        p = ans.noul if isinstance(ans, NoulAnswer) else float("nan")
        self.warmup_s = round(time.perf_counter() - t0, 3)
        print(
            f"Warm-up judgment: p(greeting)={p:.4f} in {self.warmup_s:.1f}s "
            f"(template {self.session.served.value}, build {self.build}, GPU {self.gpu})",
            flush=True,
        )

    def status(self) -> dict[str, Any]:
        """Return the facts that the page header shows.

        Returns:
            Model, template, GPU, build, router, warm-up time and gallery.
        """
        return {
            "model": self.session.model,
            "template": self.session.served.value,
            "gpu": self.gpu,
            "build": self.build,
            "router": self.settings.base_url,
            "warmup_s": self.warmup_s,
            "receipts": [
                {"id": rid, "file": self.receipts[rid]["image"]["file_name"]}
                for rid in RECEIPT_IDS
                if rid in self.receipts
            ],
            "r01_total": self.receipts["R01"]["annotated_total"],
            "license": self.manifest.get("license", "CC-BY-4.0"),
        }

    def _judge_captured(
        self,
        state: str,
        questions: Mapping[str, Question],
        media: tuple[ImageInput, ...] = (),
    ) -> tuple[
        JudgmentResponse,
        float,
        int,
        list[dict[str, Any]],
        list[tuple[Any, Any]],
        list[int | None],
    ]:
        """Run one judgment under the lock with the capture taps on.

        Args:
            state: Claim or customer message.
            questions: Question names to typed questions.
            media: Images that condition the judgment.

        Returns:
            The response, seconds, router calls, captured exchanges, scoring
            pairs and the text token count of each ``/completion`` prompt.
        """
        with self.lock:
            t0 = time.perf_counter()
            n0 = len(self.http_log)
            self.capture.start()
            try:
                resp = self.session.port.judge(
                    state, questions, self.session.model, media=media
                )
            finally:
                entries, scored = self.capture.stop()
            elapsed = round(time.perf_counter() - t0, 3)
            calls = len(self.http_log) - n0
            marker = self.session.capability.marker
            text_counts = [
                bts.count_text_tokens(
                    self.client, self.session.model, bts.prompt_string(e["req"]), marker
                )
                for e in entries
                if e["path"] == "/completion"
            ]
        return resp, elapsed, calls, entries, scored, text_counts

    def _bts(
        self,
        entries: list[dict[str, Any]],
        scored: list[tuple[Any, Any]],
        text_counts: list[int | None],
        index: int,
        normalized: dict[str, float],
        image_desc: str | None,
    ) -> dict[str, Any]:
        """Build the behind-the-scenes panel for one question.

        Args:
            entries: Captured router exchanges.
            scored: Captured scoring pairs.
            text_counts: Text token count of each ``/completion`` prompt.
            index: Position of the question in scoring order.
            normalized: Answer probabilities from the typevet answer.
            image_desc: Text that replaces the media marker, or None.

        Returns:
            The panel data from ``bts.build``.
        """
        return bts.build(
            entries=entries,
            scored=scored,
            index=index,
            normalized=normalized,
            marker=self.session.capability.marker,
            image_desc=image_desc,
            text_tokens=text_counts[index] if index < len(text_counts) else None,
        )

    # ------------------------------------------------------------- receipts
    def _save(self, kind: str, body: dict[str, Any]) -> str:
        """Write one JSON receipt under ``typevet-receipts/``.

        Args:
            kind: ``image`` or ``text``.
            body: Request, answers, timing and panel data.

        Returns:
            The receipt file name.
        """
        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = RECEIPT_DIR / f"live-demo-{stamp}-{kind}.json"
        base = {
            "kind": f"typevet.live-ui.{kind}",
            "timestamp_utc": stamp,
            "model": self.session.model,
            "template": self.session.served.value,
            "llama_cpp_build": self.build,
            "gpu": self.gpu,
            "router": self.settings.base_url,
        }
        path.write_text(json.dumps({**base, **body}, indent=2, sort_keys=True))
        return path.name

    # ---------------------------------------------------------------- image
    def _image_from_request(self, req: dict[str, Any]) -> tuple[ImageInput, str, str]:
        """Return the gallery receipt or the upload that the request names.

        Args:
            req: Parsed request body.

        Returns:
            The image, its display name and its sha256.

        Raises:
            DemoError: When the upload is malformed or no receipt is named.
        """
        rid = req.get("receipt_id")
        upload = req.get("upload")
        if upload:
            if not isinstance(upload, dict):
                raise DemoError("upload must be an object")
            name = str(upload.get("name") or "upload")
            mime = str(upload.get("mime_type") or "")
            try:
                data = base64.b64decode(
                    str(upload.get("data_b64") or ""), validate=True
                )
            except (binascii.Error, ValueError) as exc:
                raise DemoError("upload is not valid base64") from exc
            # Typed validation first: ImageInput rejects bad mime types / empty bytes.
            image = ImageInput(data=data, mime_type=mime)
            if not data.startswith(MAGIC[mime]):
                raise DemoError(f"file content does not look like {mime}")
            return image, name, hashlib.sha256(data).hexdigest()
        if rid not in self.receipts:
            raise DemoError("pick one of the six receipts or upload an image")
        rec = self.receipts[rid]
        data = (CORD / rec["image"]["file_name"]).read_bytes()
        image = ImageInput(data=data, mime_type=rec["image"]["mime_type"])
        return image, rec["image"]["file_name"], hashlib.sha256(data).hexdigest()

    def judge_image(self, req: dict[str, Any]) -> dict[str, Any]:
        """Judge whether the image supports the claim.

        Args:
            req: Parsed request body with ``claim`` and ``receipt_id`` or
                ``upload``.

        Returns:
            The page result, or a rejection when typed validation refuses
            the image.

        Raises:
            DemoError: When the claim is empty.
        """
        claim = str(req.get("claim") or "").strip()
        n0 = len(self.http_log)
        try:
            image, name, sha = self._image_from_request(req)
        except ScoringValidationError as exc:
            return {
                "rejected": True,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "router_http_calls": len(self.http_log) - n0,
            }
        if not claim:
            raise DemoError("enter a claim to check")
        q = {
            "verdict": Choice(
                instructions=VERDICT_INSTRUCTIONS, criteria=VERDICT_CRITERIA
            )
        }
        resp, elapsed, calls, entries, scored, tcounts = self._judge_captured(
            claim, q, media=(image,)
        )
        v = resp.choices["verdict"]
        size = bts.image_size(image.data, image.mime_type)
        dims = f"{size[0]}\u00d7{size[1]}" if size else "size unknown"
        image_desc = f"image: {name}, {dims}, sha256 {sha[:12]}\u2026"
        behind = self._bts(
            entries, scored, tcounts, 0, dict(v.probabilities), image_desc
        )
        result = {
            "id": "verdict",
            "type": "choice",
            "title": VERDICT_INSTRUCTIONS,
            "options": [
                {"key": k, "label": k, "desc": d, "p": v.probabilities[k]}
                for k, d in VERDICT_CRITERIA.items()
            ],
            "winner": v.choice,
            "bts": behind,
        }
        rec = self.receipts.get(str(req.get("receipt_id")))
        saved = self._save(
            "image",
            {
                "request": {
                    "claim": claim,
                    "question": VERDICT_INSTRUCTIONS,
                    "criteria": VERDICT_CRITERIA,
                    "image_name": name,
                    "receipt_id": None if req.get("upload") else req.get("receipt_id"),
                    "manifest_total": rec["annotated_total"]
                    if rec and not req.get("upload")
                    else None,
                },
                "image": {
                    "sha256": sha,
                    "mime_type": image.mime_type,
                    "byte_count": len(image.data),
                },
                "answers": {
                    "verdict": {
                        "choice": v.choice,
                        "confidence": v.confidence,
                        "probabilities": v.probabilities,
                    }
                },
                "response_model": resp.model,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
                "timing": {"elapsed_s": elapsed, "router_http_calls": calls},
                "behind_the_scenes": {"verdict": behind},
            },
        )
        return {
            "results": [result],
            "elapsed_s": elapsed,
            "router_http_calls": calls,
            "receipt_file": saved,
            "image_name": name,
            "image_sha256": sha,
        }

    # ----------------------------------------------------------------- text
    def judge_text(self, req: dict[str, Any]) -> dict[str, Any]:
        """Ask the three typed questions about a customer message.

        Args:
            req: Parsed request body with ``message``.

        Returns:
            The page results for the three questions.

        Raises:
            DemoError: When the message is empty.
        """
        message = str(req.get("message") or "").strip()
        if not message:
            raise DemoError("enter a customer message")
        resp, elapsed, calls, entries, scored, tcounts = self._judge_captured(
            message, TEXT_QUESTIONS
        )
        a = resp.nouls["unauthorized"]
        c = resp.choices["fraud_type"]
        s = resp.scores["urgency"]
        top = max(s.probabilities, key=lambda k: s.probabilities[k])
        # Questions are scored in dict order, one /completion each.
        behind = {
            "unauthorized": self._bts(
                entries,
                scored,
                tcounts,
                0,
                {"true": a.noul, "false": 1.0 - a.noul},
                None,
            ),
            "fraud_type": self._bts(
                entries, scored, tcounts, 1, dict(c.probabilities), None
            ),
            "urgency": self._bts(
                entries,
                scored,
                tcounts,
                2,
                {str(k): v for k, v in s.probabilities.items()},
                None,
            ),
        }
        results = [
            {
                "id": "unauthorized",
                "type": "yes/no",
                "title": TEXT_QUESTIONS["unauthorized"].instructions,
                "options": [
                    {"key": "yes", "label": "yes", "p": a.noul},
                    {"key": "no", "label": "no", "p": 1.0 - a.noul},
                ],
                "winner": "yes" if a.noul >= HALF else "no",
                "bts": behind["unauthorized"],
            },
            {
                "id": "fraud_type",
                "type": "pick one",
                "title": TEXT_QUESTIONS["fraud_type"].instructions,
                "options": [
                    {"key": k, "label": k, "desc": d, "p": c.probabilities[k]}
                    for k, d in FRAUD_CRITERIA.items()
                ],
                "winner": c.choice,
                "bts": behind["fraud_type"],
            },
            {
                "id": "urgency",
                "type": "score 0-3",
                "title": TEXT_QUESTIONS["urgency"].instructions,
                "options": [
                    {
                        "key": str(lv),
                        "label": f"{lv}  {s.legend[lv]}",
                        "p": s.probabilities[lv],
                    }
                    for lv in sorted(s.probabilities)
                ],
                "winner": str(top),
                "expected": s.score,
                "max": len(URGENCY_LEVELS) - 1,
                "bts": behind["urgency"],
            },
        ]
        saved = self._save(
            "text",
            {
                "request": {
                    "message": message,
                    "questions": {
                        "unauthorized": {
                            "type": "noul",
                            "instructions": TEXT_QUESTIONS["unauthorized"].instructions,
                        },
                        "fraud_type": {"type": "choice", "criteria": FRAUD_CRITERIA},
                        "urgency": {"type": "score", "levels": URGENCY_LEVELS},
                    },
                },
                "answers": {
                    "unauthorized": {"p_yes": a.noul},
                    "fraud_type": {
                        "choice": c.choice,
                        "confidence": c.confidence,
                        "probabilities": c.probabilities,
                    },
                    "urgency": {
                        "expected_value": s.score,
                        "confidence": s.confidence,
                        "probabilities": {
                            str(k): v for k, v in s.probabilities.items()
                        },
                    },
                },
                "response_model": resp.model,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
                "timing": {"elapsed_s": elapsed, "router_http_calls": calls},
                "behind_the_scenes": behind,
            },
        )
        return {
            "results": results,
            "elapsed_s": elapsed,
            "router_http_calls": calls,
            "receipt_file": saved,
        }


LOG = logging.getLogger(__name__)
THUMBS: dict[str, bytes] = {}
THUMB_LOCK = threading.Lock()


def thumbnail(rid: str) -> bytes | None:
    """Return a cached 360 px wide JPEG of a gallery receipt (display only).

    Pillow builds the thumbnail in memory. The model always gets the original
    PNG bytes.

    Args:
        rid: Gallery receipt id, such as ``R01``.

    Returns:
        The JPEG bytes, or None when Pillow cannot read the image.
    """
    with THUMB_LOCK:
        if rid not in THUMBS:
            try:
                with Image.open(CORD / f"{rid}.png") as img:
                    shown = ImageOps.exif_transpose(img).convert("RGB")
                    height = max(1, round(shown.height * THUMB_WIDTH / shown.width))
                    small = shown.resize((THUMB_WIDTH, height))
                    out = BytesIO()
                    small.save(out, format="JPEG", quality=82)
            except OSError:
                return None
            THUMBS[rid] = out.getvalue()
        return THUMBS[rid]


def one_line(exc: BaseException) -> str:
    """Return the error message on one line, at most 300 characters.

    Args:
        exc: Error to show.

    Returns:
        The shortened message, or the error type name.
    """
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:300]


class Handler(BaseHTTPRequestHandler):
    """Serves the page, the gallery images and the two judgment endpoints.

    Attributes:
        demo (Demo): The warm demo that ``main`` sets before serving.
        server_version (str): Value of the ``Server`` header.

    Examples:
        ```python
        Handler.demo = Demo()
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
        ```
    """

    demo: ClassVar[Demo]
    server_version = "typevet-live-ui/1"

    def log_message(self, format: str, *args: Any) -> None:
        """Write one access log line to stderr with the local time.

        The base class passes a printf-style format and the values that fill it.
        """
        stamp = datetime.now(UTC).astimezone()
        sys.stderr.write(f"[{stamp:%H:%M:%S}] {format % args}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        """Send one response with a cache header.

        Args:
            code: HTTP status.
            body: Response bytes.
            ctype: Content type.
        """
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "no-store" if ctype.startswith("application/json") else "max-age=3600",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        """Send one JSON response.

        Args:
            code: HTTP status.
            obj: Value to encode.
        """
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        """Serve the page, the status, a gallery image or a thumbnail."""
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(
                200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif path == "/api/status":
            self._json(200, self.demo.status())
        elif path.startswith("/img/"):
            name = path[len("/img/") :]
            if name.removesuffix(".png") in RECEIPT_IDS and name.endswith(".png"):
                self._send(200, (CORD / name).read_bytes(), "image/png")
            else:
                self._json(404, {"error": "not found"})
        elif path.startswith("/thumb/"):
            name = path[len("/thumb/") :]
            if name.removesuffix(".jpg") in RECEIPT_IDS and name.endswith(".jpg"):
                thumb = thumbnail(name.removesuffix(".jpg"))
                if thumb is not None:
                    self._send(200, thumb, "image/jpeg")
                else:
                    self._send(
                        200,
                        (CORD / f"{name.removesuffix('.jpg')}.png").read_bytes(),
                        "image/png",
                    )
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def _read_json(self) -> dict[str, Any]:
        """Read and parse the request body.

        Returns:
            The JSON object.

        Raises:
            DemoError: When the body is missing, too large, not JSON or not an
                object.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise DemoError("request body missing or too large")
        try:
            req = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DemoError("request body is not JSON") from exc
        if not isinstance(req, dict):
            raise DemoError("request body must be a JSON object")
        return req

    def do_POST(self) -> None:
        """Run an image or a text judgment and send the result as JSON."""
        path = self.path.split("?", 1)[0]
        try:
            req = self._read_json()
            if path == "/api/image":
                out = self.demo.judge_image(req)
                self._json(422 if out.get("rejected") else 200, out)
            elif path == "/api/text":
                self._json(200, self.demo.judge_text(req))
            else:
                self._json(404, {"error": "not found"})
        except DemoError as exc:
            self._json(400, {"error": one_line(exc)})
        except (ScoringValidationError, JudgmentValidationError) as exc:
            self._json(422, {"error": f"{type(exc).__name__}: {one_line(exc)}"})
        except (GenerationError, httpx.HTTPError) as exc:
            self._json(502, {"error": f"model call failed: {one_line(exc)}"})
        except Exception as exc:  # demo server: never leak a stack trace to the page
            LOG.exception("unexpected error")
            self._json(500, {"error": f"unexpected error: {one_line(exc)}"})


def main() -> int:
    """Warm the session, then serve the page until Ctrl+C.

    Returns:
        0 after a clean stop, 1 when startup fails.
    """
    try:
        demo = Demo()
    except (httpx.HTTPError, GenerationError, ValueError, OSError) as exc:
        print(f"startup failed: {one_line(exc)}", file=sys.stderr)
        return 1
    Handler.demo = demo
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ready  http://{HOST}:{PORT}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        demo.stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
