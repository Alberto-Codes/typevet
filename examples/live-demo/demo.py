"""Warm judgment session of the live demo web page.

``Demo`` holds one warm typevet judgment session on the llama.cpp router and
serializes the judgments with a lock, so the behind-the-scenes captures do not
mix. Each judgment writes one JSON receipt under ``typevet-receipts/``.
``judge.py`` builds the page results and the receipt bodies; ``bts.py``
builds the behind-the-scenes panels.

``server.py`` and ``routes.py`` import this module from the same directory.

Examples:
    ```python
    demo = Demo()
    print(demo.status()["model"])
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
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bts
import httpx
import judge

from typevet.adapters.inbound import load_llama_settings
from typevet.domain import (
    ImageInput,
    JudgmentResponse,
    Noul,
    NoulAnswer,
    Question,
    ScoringValidationError,
)
from typevet.runtime import GemmaNativeVisionSession, open_gemma_native_vision_judgment

DEFAULT_ROUTER = "http://127.0.0.1:8090"
DEFAULT_MODEL = "gemma-4-31b-kv9-q4km-mm"
DEFAULT_TIMEOUT_S = "900"
REPO = Path.cwd()
RECEIPT_DIR = REPO / "typevet-receipts"
CORD = REPO / "tests" / "fixtures" / "cord" / "expense_smoke"
RECEIPT_IDS = [f"R0{i}" for i in range(1, 7)]
MAGIC = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),
}
WARM_QUESTIONS: dict[str, Question] = {
    "greeting": Noul(
        instructions="Is this text a greeting?",
        criteria={"true": "It is a greeting", "false": "It is not a greeting"},
    )
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


def _router_env() -> dict[str, str]:
    """Return the environment with the demo defaults for the router settings.

    Returns:
        A copy of the environment with the router URL, model and timeout set.
    """
    env = dict(os.environ)
    env.setdefault("TYPEVET_LLAMA__BASE_URL", DEFAULT_ROUTER)
    env.setdefault("TYPEVET_LLAMA__MULTIMODAL_MODEL", DEFAULT_MODEL)
    env.setdefault("TYPEVET_LLAMA__TIMEOUT", DEFAULT_TIMEOUT_S)
    return env


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
        self.settings = load_llama_settings(_router_env())
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
        self.warmup_s = self._warm_up(t0)

    def _warm_up(self, t0: float) -> float:
        """Judge one greeting so the first page request does not wait.

        Args:
            t0: ``time.perf_counter`` value from before the session opened.

        Returns:
            Seconds to open the session and judge once.
        """
        warm = self.session.port.judge(
            "Hello there!", WARM_QUESTIONS, self.session.model
        )
        ans = warm.answers["greeting"]
        p = ans.noul if isinstance(ans, NoulAnswer) else float("nan")
        warmup_s = round(time.perf_counter() - t0, 3)
        print(
            f"Warm-up judgment: p(greeting)={p:.4f} in {warmup_s:.1f}s "
            f"(template {self.session.served.value}, build {self.build}, GPU {self.gpu})",
            flush=True,
        )
        return warmup_s

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
        resp, elapsed, calls, entries, scored, tcounts = self._judge_captured(
            claim, judge.VERDICT_QUESTIONS, media=(image,)
        )
        v = resp.choices["verdict"]
        desc = judge.image_desc(image, name, sha)
        behind = self._bts(entries, scored, tcounts, 0, dict(v.probabilities), desc)
        result = judge.image_result(resp, behind)
        rec = self.receipts.get(str(req.get("receipt_id")))
        request = judge.image_request(claim, req, rec, name)
        saved = self._save(
            "image",
            judge.image_receipt(request, image, sha, resp, elapsed, calls, behind),
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
            message, judge.TEXT_QUESTIONS
        )

        def panel(index: int, normalized: dict[str, float]) -> dict[str, Any]:
            """Build the panel of one text question.

            Args:
                index: Position of the question in scoring order.
                normalized: Answer probabilities from the typevet answer.

            Returns:
                The panel data from ``bts.build``.
            """
            return self._bts(entries, scored, tcounts, index, normalized, None)

        behind = judge.text_behind(resp, panel)
        results = judge.text_results(resp, behind)
        saved = self._save(
            "text", judge.text_receipt(message, resp, elapsed, calls, behind)
        )
        return {
            "results": results,
            "elapsed_s": elapsed,
            "router_http_calls": calls,
            "receipt_file": saved,
        }
