"""Behind-the-scenes capture for the live demo web page.

Two taps, neither of which changes what typevet sends or how it scores:

* httpx event hooks on the session's client record every router exchange
  (method, path, JSON body, JSON reply, wall time).
* ``scoring_port_wrapper`` (a documented hook of
  ``open_gemma_native_vision_judgment``) wraps the real
  ``LlamaCppCandidateScoringAdapter`` and records each
  ``CandidateScoringRequest`` / ``CandidateScoringResult`` pair. The raw
  per-candidate logprob therefore comes from what typevet itself extracted
  from ``completion_probabilities[0].top_logprobs``; the normalized value
  comes from typevet's answer object (the same numbers the bars draw).
  The demo only recomputes the softmax as a cross-check.

``server.py`` imports this module from the same directory.

Examples:
    ```python
    capture = Capture()
    capture.start()
    entries, scored = capture.stop()
    ```

See Also:
    - [typevet.runtime.open_gemma_native_vision_judgment][]: ``scoring_port_wrapper``.
    - examples/live-demo/README.md: How to run the live demo.
"""

from __future__ import annotations

import json
import math
import struct
import time
from typing import Any

import httpx

from typevet.adapters.outbound.gemma import (
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
)

TOP_N = 8
MAX_LIST = 12
PREFILL_NOTE = "typevet writes this; the model's next token must be an answer"
PREFILL_SOURCE = (
    "src/typevet/adapters/outbound/gemma/served_template.py "
    "(GEMMA4_NO_THINKING_PREFILL), appended by compose_media_scoring_prefix in "
    "src/typevet/adapters/outbound/gemma/scoring_prefix.py"
)
JPEG_MARKER = 0xFF
JPEG_SOF_FIRST = 0xC0
JPEG_SOF_LAST = 0xCF
JPEG_NOT_SOF = (0xC4, 0xC8, 0xCC)


def _png_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from a PNG ``IHDR`` chunk.

    Args:
        data: Image bytes.

    Returns:
        The size, or None when the header is not ``IHDR``.
    """
    if data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from the first JPEG start-of-frame segment.

    Args:
        data: Image bytes.

    Returns:
        The size, or None when no start-of-frame segment is found.
    """
    i = 2
    while i + 9 < len(data):
        if data[i] != JPEG_MARKER:
            i += 1
            continue
        m = data[i + 1]
        seg = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if JPEG_SOF_FIRST <= m <= JPEG_SOF_LAST and m not in JPEG_NOT_SOF:
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        i += 2 + seg
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from a WebP ``VP8X``, ``VP8`` or ``VP8L`` chunk.

    Args:
        data: Image bytes.

    Returns:
        The size, or None for another chunk kind.
    """
    kind = data[12:16]
    if kind == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if kind == b"VP8 ":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        b = int.from_bytes(data[21:25], "little")
        return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    return None


def image_size(data: bytes, mime: str) -> tuple[int, int] | None:
    """Return (width, height) for PNG, JPEG or WebP bytes, else None.

    Args:
        data: Image bytes.
        mime: Image mime type.

    Returns:
        The size, or None when the type or the header is not known.
    """
    readers = {
        "image/png": _png_size,
        "image/jpeg": _jpeg_size,
        "image/webp": _webp_size,
    }
    reader = readers.get(mime)
    try:
        return reader(data) if reader is not None else None
    except (struct.error, IndexError):
        return None


def _trim(obj: Any) -> Any:
    """Shorten long lists recursively so the raw JSON stays readable.

    Args:
        obj: Parsed JSON value.

    Returns:
        The value with each list cut to ``MAX_LIST`` items and a note.
    """
    if isinstance(obj, list):
        head = [_trim(x) for x in obj[:MAX_LIST]]
        if len(obj) > MAX_LIST:
            head.append(f"... (+{len(obj) - MAX_LIST} more, trimmed by demo)")
        return head
    if isinstance(obj, dict):
        return {k: _trim(v) for k, v in obj.items()}
    return obj


def _sanitize_request(body: Any) -> Any:
    """Replace base64 image payloads with a length placeholder.

    Args:
        body: Parsed ``/completion`` request body.

    Returns:
        The body with each image payload replaced by its length.
    """
    if isinstance(body, dict) and isinstance(body.get("prompt"), dict):
        prompt = dict(body["prompt"])
        data = prompt.get("multimodal_data")
        if isinstance(data, list):
            prompt["multimodal_data"] = [
                f"<base64 image data elided: {len(d)} chars>" for d in data
            ]
        body = {**body, "prompt": prompt}
    return body


def _trim_completion(resp: Any) -> Any:
    """Keep only the top entries of the huge n_vocab logprob list.

    Args:
        resp: Parsed ``/completion`` reply.

    Returns:
        The reply with the logprob lists cut to ``TOP_N`` entries.
    """
    if not isinstance(resp, dict):
        return resp
    out = dict(resp)
    cps = out.get("completion_probabilities")
    if isinstance(cps, list) and cps and isinstance(cps[0], dict):
        first = dict(cps[0])
        tops = first.get("top_logprobs")
        if isinstance(tops, list):
            first["top_logprobs"] = tops[:TOP_N] + (
                [f"... (+{len(tops) - TOP_N} more vocabulary entries, trimmed by demo)"]
                if len(tops) > TOP_N
                else []
            )
        out["completion_probabilities"] = [first] + (
            [f"... (+{len(cps) - 1} more)"] if len(cps) > 1 else []
        )
    return _trim(out)


class CapturingScoringPort:
    """Pass-through CandidateScoringPort that records request/result pairs.

    Attributes:
        _inner (Any): The real scoring port.
        _capture (Capture): Capture that receives the pairs.

    Examples:
        ```python
        port = Capture().wrap(inner_scoring_port)
        ```
    """

    def __init__(self, inner: Any, capture: Capture) -> None:
        """Store the real port and the capture.

        Args:
            inner: The real scoring port.
            capture: Capture that receives the pairs.
        """
        self._inner = inner
        self._capture = capture

    def score_candidates(self, request: Any) -> Any:
        """Score with the real port and record the pair while capture is on.

        Args:
            request: Candidate scoring request.

        Returns:
            The result from the real port, unchanged.
        """
        result = self._inner.score_candidates(request)
        if self._capture.active is not None:
            self._capture.scored.append((request, result))
        return result


class Capture:
    """Collects router exchanges while ``active`` is a list (under the demo lock).

    Attributes:
        active (list[dict[str, Any]] | None): Exchanges of the current
            judgment, or None when capture is off.
        scored (list[tuple[Any, Any]]): Scoring request and result pairs.

    Examples:
        ```python
        capture = Capture()
        capture.install(client)
        ```
    """

    def __init__(self) -> None:
        """Start with capture off."""
        self.active: list[dict[str, Any]] | None = None
        self.scored: list[tuple[Any, Any]] = []
        self._t0: dict[int, float] = {}

    def install(self, client: httpx.Client) -> None:
        """Add the request and response hooks to ``client``.

        Args:
            client: Session HTTP client.
        """
        client.event_hooks["request"].append(self._on_request)
        client.event_hooks["response"].append(self._on_response)

    def wrap(self, inner: Any) -> CapturingScoringPort:
        """Wrap the real scoring port; use as ``scoring_port_wrapper``.

        Args:
            inner: The real scoring port.

        Returns:
            A pass-through port that records each pair.
        """
        return CapturingScoringPort(inner, self)

    def start(self) -> None:
        """Turn capture on with empty lists."""
        self.active = []
        self.scored = []

    def stop(self) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
        """Turn capture off and return what it recorded.

        Returns:
            The router exchanges and the scoring pairs.
        """
        entries, scored = self.active or [], self.scored
        self.active, self.scored = None, []
        return entries, scored

    def _on_request(self, request: httpx.Request) -> None:
        if self.active is not None:
            self._t0[id(request)] = time.perf_counter()

    def _on_response(self, response: httpx.Response) -> None:
        if self.active is None:
            return
        response.read()
        req = response.request
        t0 = self._t0.pop(id(req), None)
        ms = round((time.perf_counter() - t0) * 1000, 1) if t0 is not None else None
        try:
            req_json = json.loads(req.content) if req.content else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            req_json = None
        try:
            resp_json = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            resp_json = None
        self.active.append(
            {
                "method": req.method,
                "path": req.url.path,
                "status": response.status_code,
                "ms": ms,
                "req": req_json,
                "resp": resp_json,
            }
        )


def prompt_string(body: Any) -> str:
    """Return the prompt text of a ``/completion`` request body.

    Args:
        body: Parsed request body.

    Returns:
        The prompt string, or an empty string.
    """
    p = body.get("prompt") if isinstance(body, dict) else None
    if isinstance(p, dict):
        return str(p.get("prompt_string", ""))
    return str(p or "")


def call_list(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per router call, for the "other calls" list.

    Args:
        entries: Router exchanges from ``Capture.stop``.

    Returns:
        One summary row for each exchange.
    """
    out = []
    for i, e in enumerate(entries):
        note = ""
        if e["path"] == "/tokenize" and isinstance(e["req"], dict):
            toks = (
                (e["resp"] or {}).get("tokens") if isinstance(e["resp"], dict) else None
            )
            note = f"{json.dumps(e['req'].get('content'))} -> {toks}"
        elif e["path"] == "/completion" and isinstance(e["resp"], dict):
            note = (
                f"tokens_evaluated={e['resp'].get('tokens_evaluated')} "
                f"tokens_predicted={e['resp'].get('tokens_predicted')}"
            )
        out.append(
            {
                "i": i,
                "method": e["method"],
                "path": e["path"],
                "status": e["status"],
                "ms": e["ms"],
                "note": note,
            }
        )
    return out


def build(
    *,
    entries: list[dict[str, Any]],
    scored: list[tuple[Any, Any]],
    index: int,
    normalized: dict[str, float],
    marker: str,
    image_desc: str | None,
    text_tokens: int | None,
) -> dict[str, Any]:
    """Assemble the panel data for the ``index``-th scored question.

    Args:
        entries: Router exchanges from ``Capture.stop``.
        scored: Scoring pairs from ``Capture.stop``.
        index: Position of the question in scoring order.
        normalized: Answer probabilities from the typevet answer.
        marker: Media marker of the session.
        image_desc: Text that replaces the media marker, or None for text.
        text_tokens: Token count of the prompt text, or None.

    Returns:
        The behind-the-scenes panel data.
    """
    completions = [e for e in entries if e["path"] == "/completion"]
    comp = completions[index]
    sreq, sres = scored[index]
    prompt = prompt_string(comp["req"])
    # Media marker -> readable placeholder (never the base64).
    parts = prompt.split(marker) if marker and image_desc else [prompt]
    shown = f"⟨{image_desc}⟩".join(parts) if len(parts) > 1 else prompt
    prefill = (
        GEMMA4_NO_THINKING_PREFILL if shown.endswith(GEMMA4_NO_THINKING_PREFILL) else ""
    )
    body = shown[: len(shown) - len(prefill)] if prefill else shown
    opener = GEMMA4_MODEL_TURN_HEADER if body.endswith(GEMMA4_MODEL_TURN_HEADER) else ""
    body = body[: len(body) - len(opener)] if opener else body

    # control string for each token id, from the /tokenize calls typevet made
    control_of: dict[tuple[int, ...], str] = {}
    for e in entries:
        if (
            e["path"] == "/tokenize"
            and isinstance(e["req"], dict)
            and isinstance(e["resp"], dict)
        ):
            control_of[tuple(e["resp"].get("tokens") or ())] = str(
                e["req"].get("content")
            )

    resp = comp["resp"] if isinstance(comp["resp"], dict) else {}
    tops = []
    try:
        tops = resp["completion_probabilities"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        tops = []
    piece_of = {
        int(t["id"]): t.get("token") for t in tops if isinstance(t, dict) and "id" in t
    }

    rows: list[dict[str, Any]] = []
    lps = [c.logprob for c in sres.candidates]
    mx = max(lps)
    z = sum(math.exp(lp - mx) for lp in lps)
    for _spec, cand in zip(sreq.candidates, sres.candidates, strict=True):
        tid = tuple(cand.token_ids)
        rows.append(
            {
                "option": str(cand.label),
                "control": control_of.get(tid, "?"),
                "token_ids": list(tid),
                "token_piece": piece_of.get(tid[0]) if tid else None,
                "raw_logprob": cand.logprob,
                "raw_p": math.exp(cand.logprob),
                # yes/no labels arrive as "True"/"False"; the answer map uses lower case
                "normalized_p": normalized.get(
                    str(cand.label), normalized.get(str(cand.label).lower())
                ),
                "normalized_recomputed": math.exp(cand.logprob - mx) / z,
                "token_count": len(tid),
            }
        )
    on_menu = sum(r["raw_p"] for r in rows)
    menu_ids = {r["token_ids"][0] for r in rows if r["token_ids"]}
    top_tokens = [
        {
            "id": int(t["id"]),
            "token": t.get("token"),
            "p": math.exp(float(t["logprob"])),
            "on_menu": int(t["id"]) in menu_ids,
        }
        for t in tops[:TOP_N]
        if isinstance(t, dict)
    ]

    raw_timings = resp.get("timings")
    timings: dict[str, Any] = raw_timings if isinstance(raw_timings, dict) else {}
    prompt_tokens = resp.get("tokens_evaluated")
    image_tokens = (
        prompt_tokens - text_tokens
        if image_desc
        and isinstance(prompt_tokens, int)
        and isinstance(text_tokens, int)
        else (0 if not image_desc else None)
    )
    # Prefix-cache reuse is timings.cache_n (tokens taken from the slot cache).
    # tokens_cached is the slot's total token count after the call, not reuse.
    cached = timings.get("cache_n")
    tokens = {
        "prompt_tokens": prompt_tokens,
        "text_tokens": text_tokens,
        "image_tokens_approx": image_tokens,
        "generated_tokens": resp.get("tokens_predicted"),
        "tokens_cached": cached,
        "slot_tokens_after": resp.get("tokens_cached"),
        "cache_prompt_requested": (comp["req"] or {}).get("cache_prompt"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "wall_ms": comp["ms"],
        "n_probs_returned": len(tops),
        "source": (
            "prompt/generated/cached from the /completion reply (tokens_evaluated, "
            "tokens_predicted, timings.cache_n, timings.prompt_ms); text_tokens from one extra "
            "/tokenize of the prompt text by the demo (image marker removed, "
            "add_special=true); image = prompt - text"
        ),
    }
    return {
        "prompt": {
            "body": body,
            "opener": opener,
            "prefill": prefill,
            "prefill_note": PREFILL_NOTE,
            "prefill_source": PREFILL_SOURCE,
            "chars": len(prompt),
        },
        "candidates": rows,
        "mass": {"on_menu": on_menu, "off_menu": max(0.0, 1.0 - on_menu)},
        "top_tokens": top_tokens,
        "tokens": tokens,
        "completion": {
            "ms": comp["ms"],
            "status": comp["status"],
            "request": _trim(_sanitize_request(comp["req"])),
            "response": _trim_completion(comp["resp"]),
        },
        "calls": call_list(entries),
        "completion_call_index": entries.index(comp),
    }


def count_text_tokens(
    client: httpx.Client, model: str, prompt: str, marker: str
) -> int | None:
    """Demo-side count of the text part of a prompt (not typevet).

    Args:
        client: Session HTTP client.
        model: Model id.
        prompt: Prompt text with the media marker.
        marker: Media marker to remove before the count.

    Returns:
        The token count, or None when ``/tokenize`` fails.
    """
    try:
        r = client.post(
            "/tokenize",
            json={
                "model": model,
                "content": prompt.replace(marker, ""),
                "add_special": True,
            },
        )
        return len(r.raise_for_status().json()["tokens"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None
