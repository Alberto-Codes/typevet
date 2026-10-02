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

``demo.py`` imports this module from the same directory. ``build`` puts the
panel together from small helpers, one for each panel section. The image
size readers are in ``image_size.py`` and the raw JSON trimmers are in
``raw_json.py``.

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
import time
from typing import Any

import httpx
from raw_json import TOP_N, sanitize_request, trim, trim_completion

from typevet.adapters.outbound.gemma import (
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
)

PREFILL_NOTE = "typevet writes this; the model's next token must be an answer"
PREFILL_SOURCE = (
    "src/typevet/adapters/outbound/gemma/served_template.py "
    "(GEMMA4_NO_THINKING_PREFILL), appended by compose_media_scoring_prefix in "
    "src/typevet/adapters/outbound/gemma/scoring_prefix.py"
)


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


def _prompt_parts(
    prompt: str, marker: str, image_desc: str | None
) -> tuple[str, str, str]:
    """Split the shown prompt into body, model-turn opener and prefill.

    Args:
        prompt: Prompt text of the ``/completion`` request.
        marker: Media marker of the session.
        image_desc: Text that replaces the media marker, or None for text.

    Returns:
        The body, the opener and the prefill.
    """
    # Media marker -> readable placeholder (never the base64).
    parts = prompt.split(marker) if marker and image_desc else [prompt]
    shown = f"⟨{image_desc}⟩".join(parts) if len(parts) > 1 else prompt
    prefill = (
        GEMMA4_NO_THINKING_PREFILL if shown.endswith(GEMMA4_NO_THINKING_PREFILL) else ""
    )
    body = shown[: len(shown) - len(prefill)] if prefill else shown
    opener = GEMMA4_MODEL_TURN_HEADER if body.endswith(GEMMA4_MODEL_TURN_HEADER) else ""
    body = body[: len(body) - len(opener)] if opener else body
    return body, opener, prefill


def _control_strings(entries: list[dict[str, Any]]) -> dict[tuple[int, ...], str]:
    """Map each token id tuple to its control string.

    Args:
        entries: Router exchanges from ``Capture.stop``.

    Returns:
        The control string for each token id tuple, from the ``/tokenize``
        calls that typevet made.
    """
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
    return control_of


def _top_logprobs(resp: dict[str, Any]) -> list[Any]:
    """Return the ``top_logprobs`` list of the first completion position.

    Args:
        resp: Parsed ``/completion`` reply.

    Returns:
        The list, or an empty list when the reply has none.
    """
    tops = []
    try:
        tops = resp["completion_probabilities"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        tops = []
    return tops


def _candidate_rows(
    sreq: Any,
    sres: Any,
    control_of: dict[tuple[int, ...], str],
    tops: list[Any],
    normalized: dict[str, float],
) -> list[dict[str, Any]]:
    """Build one table row for each scored candidate.

    Args:
        sreq: Candidate scoring request.
        sres: Candidate scoring result.
        control_of: Control string for each token id tuple.
        tops: ``top_logprobs`` list of the completion.
        normalized: Answer probabilities from the typevet answer.

    Returns:
        One row for each candidate.
    """
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
    return rows


def _top_tokens(tops: list[Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ``TOP_N`` most likely next tokens, marked on or off the menu.

    Args:
        tops: ``top_logprobs`` list of the completion.
        rows: Candidate rows from ``_candidate_rows``.

    Returns:
        One entry for each of the first ``TOP_N`` tokens.
    """
    menu_ids = {r["token_ids"][0] for r in rows if r["token_ids"]}
    return [
        {
            "id": int(t["id"]),
            "token": t.get("token"),
            "p": math.exp(float(t["logprob"])),
            "on_menu": int(t["id"]) in menu_ids,
        }
        for t in tops[:TOP_N]
        if isinstance(t, dict)
    ]


def _token_facts(
    comp: dict[str, Any],
    resp: dict[str, Any],
    n_tops: int,
    image_desc: str | None,
    text_tokens: int | None,
) -> dict[str, Any]:
    """Return the token counts and timings of one ``/completion`` call.

    Args:
        comp: The ``/completion`` exchange.
        resp: Parsed ``/completion`` reply.
        n_tops: Length of the ``top_logprobs`` list.
        image_desc: Text that replaces the media marker, or None for text.
        text_tokens: Token count of the prompt text, or None.

    Returns:
        The token facts for the panel.
    """
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
    return {
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
        "n_probs_returned": n_tops,
        "source": (
            "prompt/generated/cached from the /completion reply (tokens_evaluated, "
            "tokens_predicted, timings.cache_n, timings.prompt_ms); text_tokens from one extra "
            "/tokenize of the prompt text by the demo (image marker removed, "
            "add_special=true); image = prompt - text"
        ),
    }


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
    body, opener, prefill = _prompt_parts(prompt, marker, image_desc)
    resp = comp["resp"] if isinstance(comp["resp"], dict) else {}
    tops = _top_logprobs(resp)
    rows = _candidate_rows(sreq, sres, _control_strings(entries), tops, normalized)
    on_menu = sum(r["raw_p"] for r in rows)
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
        "top_tokens": _top_tokens(tops, rows),
        "tokens": _token_facts(comp, resp, len(tops), image_desc, text_tokens),
        "completion": {
            "ms": comp["ms"],
            "status": comp["status"],
            "request": trim(sanitize_request(comp["req"])),
            "response": trim_completion(comp["resp"]),
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
