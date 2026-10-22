"""Offline-capable llama.cpp receipt helper for the judgevet live smoke (#445).

Records bounded ``/completion`` traffic, runs one ``system_one`` triple through
public ``open_judgment``, and optionally replays the first Noul request without
the empty-think suffix. Unit tests inject ``httpx.MockTransport``; the live
entry injects ``httpx.HTTPTransport(retries=0)``.

Examples:
    ```python
    from tests.live.judgevet_llama_receipt import (
        RecordingTransport,
        run_llama_bridge_receipt,
    )

    transport = RecordingTransport(httpx.MockTransport(handler), max_completions=4)
    receipt = run_llama_bridge_receipt(environ, transport=transport)
    ```

See Also:
    - [tests.live.test_judgevet_bridge_live][]: Live pytest entry
    - [typevet.adapters.inbound.backend_settings][]: Public ``open_judgment``

[i445]: https://github.com/Alberto-Codes/typevet/issues/445
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import httpx
from judgevet import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.adapters.inbound.settings import LlamaSettings, load_llama_settings
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_EMPTY_THINK_PREFILLS,
    CHATML_IM_END,
    CHATML_IM_START,
    ChatMLThinkingFraming,
)
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
)
from typevet.adapters.outbound.llama_cpp.vocabulary import n_vocab_from_models

FRAMING_ENV = "TYPEVET_LIVE_LLAMA_FRAMING"
HISTORICAL_GEMMA_FALLBACK = "gemma-4-31b-24gib-kv11-decoder"
MAX_COMPLETION_ATTEMPTS = 4
FULL_DISTRIBUTION_TOLERANCE = 1e-2
"""Matches ``_FULL_DISTRIBUTION_TOLERANCE`` in the llama.cpp scoring adapter."""

MARKERS = (CHATML_IM_START, CHATML_IM_END, "<think>", "</think>")
CONTROLS = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

STATE = (
    "Support ticket: I was charged twice for my March subscription. "
    "Please refund the duplicate payment as soon as possible."
)
_URGENCY_LEVELS = (
    "Not urgent",
    "Slightly urgent",
    "Moderately urgent",
    "Very urgent",
    "Critical",
)
QUESTIONS = {
    "duplicate_charge": Noul(
        instructions="Does the ticket report a duplicate charge?",
    ),
    "route": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, refunds and invoices.",
            "technical": "Faults, errors and outages.",
            "sales": "New plans, upgrades and quotes.",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=_URGENCY_LEVELS,
    ),
}

_STALE_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length"})


class CompletionBudgetError(RuntimeError):
    """Raised when a ``/completion`` would exceed the harness attempt budget."""


class RecordingTransport(httpx.BaseTransport):
    """Wrap a transport, bound ``/completion`` attempts, and keep first bodies.

    Attributes:
        completion_attempts (int): ``/completion`` attempts seen, including
            refusals that never reached the inner transport.
        first_completion_request (dict[str, Any] | None): JSON body of the
            first ``/completion`` request (no headers).
        first_completion_response (dict[str, Any] | None): JSON body of the
            first ``/completion`` response.
    """

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        max_completions: int = MAX_COMPLETION_ATTEMPTS,
    ) -> None:
        """Wrap ``inner`` and keep at most ``max_completions`` completions.

        Args:
            inner: Transport that performs the real or mock HTTP work.
            max_completions: Highest ``/completion`` attempt count allowed.
        """
        self._inner = inner
        self._max_completions = max_completions
        self.completion_attempts = 0
        self.first_completion_request: dict[str, Any] | None = None
        self.first_completion_response: dict[str, Any] | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Forward ``request``, enforcing the completion budget before I/O."""
        path = request.url.path
        is_completion = path == "/completion" or path.endswith("/completion")
        if is_completion:
            self.completion_attempts += 1
            if self.completion_attempts > self._max_completions:
                msg = (
                    f"/completion budget exceeded: attempt "
                    f"{self.completion_attempts} with max {self._max_completions}"
                )
                raise CompletionBudgetError(msg)
            body = json.loads(request.content.decode())
            if self.first_completion_request is None:
                self.first_completion_request = body
        response = self._inner.handle_request(request)
        if is_completion and self.first_completion_response is None:
            content = response.read()
            self.first_completion_response = json.loads(content.decode())
            headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower() not in _STALE_RESPONSE_HEADERS
            ]
            return httpx.Response(
                response.status_code,
                headers=headers,
                content=content,
                request=request,
            )
        return response

    def close(self) -> None:
        """Close the inner transport."""
        self._inner.close()


def resolve_live_llama_framing(
    environ: Mapping[str, str],
) -> ChatMLThinkingFraming | None:
    """Return ChatML framing only for an explicit ``chatml`` opt-in.

    Args:
        environ: Environment mapping that may hold ``TYPEVET_LIVE_LLAMA_FRAMING``.

    Returns:
        ``ChatMLThinkingFraming`` when the value is ``chatml``, else ``None``.

    Raises:
        ValueError: The variable is set to any value other than ``chatml``.
    """
    raw = environ.get(FRAMING_ENV, "").strip()
    if not raw:
        return None
    if raw == "chatml":
        return ChatMLThinkingFraming()
    msg = f"{FRAMING_ENV} must be 'chatml' or unset"
    raise ValueError(msg)


def selected_llama_model(
    settings: LlamaSettings,
    *,
    framing: ChatMLThinkingFraming | None,
) -> str:
    """Choose the live model from settings, with the historical Gemma fallback.

    Args:
        settings: Loaded llama.cpp settings.
        framing: Harness framing, or ``None`` for the unframed Gemma path.

    Returns:
        ``settings.default_model``, or ``HISTORICAL_GEMMA_FALLBACK`` when
        unframed and unset.

    Raises:
        ValueError: Framing is set and ``settings.default_model`` is missing.
    """
    if framing is not None:
        if not settings.default_model:
            msg = (
                "settings.default_model (TYPEVET_LLAMA__DEFAULT_MODEL) "
                "is required when framing is set"
            )
            raise ValueError(msg)
        return settings.default_model
    if settings.default_model:
        return settings.default_model
    return HISTORICAL_GEMMA_FALLBACK


def strip_empty_think_suffix(prompt: str) -> str:
    """Remove exactly one known empty-think suffix from ``prompt``.

    Args:
        prompt: The first scored ``/completion`` prompt.

    Returns:
        Prompt that ends at the ChatML assistant header.

    Raises:
        ValueError: The prompt does not end with a known empty-think prefill.
    """
    for prefill in CHATML_EMPTY_THINK_PREFILLS:
        if prompt.endswith(prefill):
            stripped = prompt[: -len(prefill)]
            if not stripped.endswith(CHATML_ASSISTANT_HEADER):
                msg = "stripped prompt must end with the ChatML assistant header"
                raise ValueError(msg)
            return stripped
    msg = "prompt does not end with a known CHATML_EMPTY_THINK_PREFILLS suffix"
    raise ValueError(msg)


def top_logprobs_from_completion(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every ``top_logprobs`` entry from the first completion slot.

    Args:
        payload: Parsed ``/completion`` JSON body.

    Returns:
        The full top-N list (not a truncated summary).
    """
    completion_probabilities = payload["completion_probabilities"]
    first = completion_probabilities[0]
    top = first["top_logprobs"]
    if not isinstance(top, list):
        msg = "llama.cpp top_logprobs must be a list"
        raise TypeError(msg)
    return top


def off_option_mass_from_top(
    top_logprobs: list[dict[str, Any]],
    *,
    candidate_ids: tuple[int, ...],
    n_vocab: int,
) -> float:
    """Compute off-option mass with the production completeness rules.

    Args:
        top_logprobs: Full vocabulary entries from ``/completion``.
        candidate_ids: Token ids for the scored options.
        n_vocab: Expected vocabulary size from the catalog.

    Returns:
        Off-option mass in ``[0, 1]``.

    Raises:
        AssertionError: The distribution is incomplete, duplicated, non-finite,
            or missing a candidate id.
        TypeError: A token id or logprob has the wrong type.
    """
    if len(top_logprobs) != n_vocab:
        msg = f"expected {n_vocab} top_logprobs entries, got {len(top_logprobs)}"
        raise AssertionError(msg)
    token_logprobs: dict[int, float] = {}
    for entry in top_logprobs:
        token_id = entry["id"]
        logprob = entry["logprob"]
        if not isinstance(token_id, int):
            msg = f"non-int token id: {token_id!r}"
            raise TypeError(msg)
        if token_id in token_logprobs:
            msg = f"duplicate token id in top_logprobs: {token_id!r}"
            raise AssertionError(msg)
        if not isinstance(logprob, int | float) or not math.isfinite(logprob):
            msg = f"non-finite logprob for token {token_id}: {logprob!r}"
            raise AssertionError(msg)
        token_logprobs[token_id] = float(logprob)
    returned_mass = math.fsum(math.exp(value) for value in token_logprobs.values())
    if abs(returned_mass - 1.0) > FULL_DISTRIBUTION_TOLERANCE:
        msg = (
            f"returned mass {returned_mass!r} is outside production tolerance "
            f"{FULL_DISTRIBUTION_TOLERANCE!r}"
        )
        raise AssertionError(msg)
    for candidate_id in candidate_ids:
        if candidate_id not in token_logprobs:
            msg = f"missing candidate token id {candidate_id} in top_logprobs"
            raise AssertionError(msg)
    candidate_mass = math.fsum(
        math.exp(token_logprobs[candidate_id]) for candidate_id in candidate_ids
    )
    return min(1.0, max(0.0, 1.0 - candidate_mass))


def scrub_secrets(value: Any, secrets: tuple[str, ...]) -> Any:
    """Return ``value`` with configured secret substrings replaced by ``***``."""
    active = tuple(secret for secret in secrets if secret)
    if not active:
        return value
    if isinstance(value, str):
        scrubbed = value
        for secret in active:
            scrubbed = scrubbed.replace(secret, "***")
        return scrubbed
    if isinstance(value, list):
        return [scrub_secrets(item, active) for item in value]
    if isinstance(value, dict):
        return {key: scrub_secrets(item, active) for key, item in value.items()}
    return value


def _tokenize(client: httpx.Client, model: str, content: str) -> tuple[int, ...]:
    """Tokenize ``content`` with ``add_special=False``."""
    response = client.post(
        "/tokenize",
        json={"model": model, "content": content, "add_special": False},
    )
    tokens = response.raise_for_status().json()["tokens"]
    return tuple(int(token) for token in tokens)


def _answers_summary(response: Any) -> dict[str, object]:
    """Build the typed answer summary for the receipt JSON."""
    noul = response.answers["duplicate_charge"]
    choice = response.answers["route"]
    score = response.answers["urgency"]
    assert isinstance(noul, NoulAnswer)
    assert isinstance(choice, ChoiceAnswer)
    assert isinstance(score, ScoreAnswer)
    true_p = float(noul.noul)
    return {
        "duplicate_charge": {
            "type": "noul",
            "noul": noul.noul,
            "probabilities": {"0": 1.0 - true_p, "1": true_p},
        },
        "route": {
            "type": "choice",
            "choice": choice.choice,
            "confidence": choice.confidence,
            "probabilities": choice.probabilities,
        },
        "urgency": {
            "type": "score",
            "score": score.score,
            "confidence": score.confidence,
            "legend": {str(k): v for k, v in score.legend.items()},
            "probabilities": {str(k): v for k, v in score.probabilities.items()},
        },
    }


def _catalog_n_vocab(client: httpx.Client, model: str) -> int | None:
    """Read ``meta.n_vocab`` for ``model`` from ``/v1/models``."""
    payload = client.get("/v1/models").raise_for_status().json()
    return n_vocab_from_models(payload, model)


def _served_entries(client: httpx.Client, model: str) -> list[dict[str, object]]:
    """Return ``/v1/models`` entries that match ``model``."""
    body = client.get("/v1/models").raise_for_status().json()
    return [
        {k: item.get(k) for k in ("id", "root", "max_model_len") if k in item}
        for item in body["data"]
        if item.get("id") == model
    ]


def _record_chatml_tokens(
    client: httpx.Client, model: str
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Tokenize ChatML markers and the 36 control labels before scoring."""
    marker_token_ids = {
        marker: list(_tokenize(client, model, marker)) for marker in MARKERS
    }
    control_token_ids: dict[str, list[int]] = {}
    for control in CONTROLS:
        tokens = _tokenize(client, model, control)
        assert len(tokens) == 1, f"control {control!r} tokenized to {tokens!r}"
        control_token_ids[control] = list(tokens)
    return marker_token_ids, control_token_ids


def _off_option_masses(
    response: Any,
    questions: Mapping[str, Any],
    *,
    framing: ChatMLThinkingFraming | None,
) -> dict[str, float | None]:
    """Collect per-answer off-option masses; framed answers require finite mass."""
    masses: dict[str, float | None] = {}
    for name in questions:
        entry = response.receipts.get(name)
        mass = None if entry is None else entry.get("off_option_mass")
        if framing is not None:
            assert mass is not None, f"missing off_option_mass for {name}"
            assert isinstance(mass, int | float)
            assert math.isfinite(mass) and 0.0 <= float(mass) <= 1.0
            masses[name] = float(mass)
        else:
            masses[name] = float(mass) if isinstance(mass, int | float) else None
    return masses


def _record_failure(
    receipt: dict[str, object],
    exc: BaseException,
    *,
    attempts: int | None,
) -> None:
    """Attach failure metadata without inventing a passing outcome."""
    receipt["failure"] = {
        "type": type(exc).__name__,
        "reason": str(exc),
        "completion_attempts": attempts,
    }


def _run_noul_control(
    client: httpx.Client,
    recorder: RecordingTransport,
    *,
    n_vocab: int,
    control_token_ids: Mapping[str, list[int]],
    original_mass: float,
    receipt: dict[str, object],
) -> dict[str, object]:
    """Replay the first completion without the empty-think suffix."""
    assert recorder.first_completion_request is not None
    control_body = deepcopy(recorder.first_completion_request)
    original_prompt = control_body["prompt"]
    assert isinstance(original_prompt, str)
    stripped = strip_empty_think_suffix(original_prompt)
    control_body["prompt"] = stripped
    control_payload = client.post("/completion", json=control_body).raise_for_status()
    payload = control_payload.json()
    false_id = control_token_ids["0"][0]
    true_id = control_token_ids["1"][0]
    control: dict[str, object] = {
        "prompt": stripped,
        "scored_prompt": original_prompt,
        "prompt_delta": original_prompt[len(stripped) :],
        "request": control_body,
        "response": payload,
        "original_off_option_mass": original_mass,
        "candidate_ids": {"0": false_id, "1": true_id},
    }
    receipt["control"] = control
    control_top = top_logprobs_from_completion(payload)
    control_mass = off_option_mass_from_top(
        control_top,
        candidate_ids=(false_id, true_id),
        n_vocab=n_vocab,
    )
    control["off_option_mass"] = control_mass
    assert control_mass > original_mass, (
        f"control off-option mass {control_mass} must exceed original {original_mass}"
    )
    return control


def _require_framed_catalog(
    session: GemmaNativeVisionSession,
    *,
    n_vocab: int | None,
) -> int:
    """Refuse unsupported templates or a missing catalog size before scoring."""
    if session.served.value != "degraded_chatml":
        msg = f"framed harness requires degraded_chatml; got {session.served.value}"
        raise ValueError(msg)
    assert n_vocab is not None, (
        "catalog meta.n_vocab is required for framed harness before scoring"
    )
    return n_vocab


def _score_and_control(
    *,
    session: GemmaNativeVisionSession,
    model: str,
    framing: ChatMLThinkingFraming | None,
    recorder: RecordingTransport,
    state: str,
    ask: Mapping[str, Any],
    out: dict[str, object],
) -> None:
    """Populate ``out`` from one opened judgment session."""
    client = session.client
    props = client.get("/props", params={"model": model}).raise_for_status().json()
    n_vocab = _catalog_n_vocab(client, model)
    out["served"] = _served_entries(client, model)
    out["server"] = {
        "model_alias": props.get("model_alias"),
        "build_info": props.get("build_info"),
        "served_template_class": session.served.value,
        "n_vocab": n_vocab,
    }
    control_token_ids: dict[str, list[int]] = {}
    framed_n_vocab: int | None = None
    if framing is not None:
        framed_n_vocab = _require_framed_catalog(session, n_vocab=n_vocab)
        markers, control_token_ids = _record_chatml_tokens(client, model)
        out["marker_token_ids"] = markers
        out["control_token_ids"] = control_token_ids

    port = TypevetSystemOnePort(session.port)
    started = time.perf_counter()
    response = port.system_one(state, ask, model)
    out["wall_seconds"] = round(time.perf_counter() - started, 3)
    out["model"] = response.model
    out["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    out["answers"] = _answers_summary(response)
    out["receipts"] = dict(response.receipts)
    masses = _off_option_masses(response, ask, framing=framing)
    out["off_option_masses"] = masses
    assert recorder.first_completion_response is not None
    first_top = top_logprobs_from_completion(recorder.first_completion_response)
    out["first_top_n"] = first_top
    out["first_top_n_count"] = len(first_top)
    if framing is not None:
        assert framed_n_vocab is not None
        original = masses["duplicate_charge"]
        assert isinstance(original, float)
        _run_noul_control(
            client,
            recorder,
            n_vocab=framed_n_vocab,
            control_token_ids=control_token_ids,
            original_mass=original,
            receipt=out,
        )
    else:
        out["control"] = None


def run_llama_bridge_receipt(
    environ: Mapping[str, str],
    *,
    transport: httpx.BaseTransport,
    state: str = STATE,
    questions: Mapping[str, Any] | None = None,
    receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    """Open public judgment, score one triple, and optionally run the control.

    Args:
        environ: Environment for settings, framing opt-in and secrets.
        transport: Injected transport (offline mock or live HTTP).
        state: Judge state text.
        questions: Judge questions; defaults to ``QUESTIONS`` (Noul first).
        receipt: Optional caller-owned mapping filled in place, including on
            failure, so the live entry can write a partial receipt.

    Returns:
        Receipt mapping with answers, masses, first top-N and optional control.

    Raises:
        ValueError: Framing or model selection is invalid before any I/O.
        AssertionError: A framed distribution or control check fails.
    """
    ask = QUESTIONS if questions is None else questions
    framing = resolve_live_llama_framing(environ)
    settings = load_llama_settings(environ)
    model = selected_llama_model(settings, framing=framing)
    session_environ = {
        **dict(environ),
        "TYPEVET_BACKEND": "llama_cpp",
        "TYPEVET_LLAMA__MULTIMODAL_MODEL": model,
    }
    if settings.default_model or framing is not None:
        session_environ["TYPEVET_LLAMA__DEFAULT_MODEL"] = model

    recorder = (
        transport
        if isinstance(transport, RecordingTransport)
        else RecordingTransport(transport, max_completions=MAX_COMPLETION_ATTEMPTS)
    )
    out: dict[str, object] = receipt if receipt is not None else {}
    out.clear()
    out.update(
        {
            "issue": 445,
            "backend": "llama_cpp",
            "requested_model": model,
        }
    )
    try:
        with open_judgment(
            session_environ, transport=recorder, framing=framing
        ) as opened:
            assert isinstance(opened, GemmaNativeVisionSession)
            _score_and_control(
                session=opened,
                model=model,
                framing=framing,
                recorder=recorder,
                state=state,
                ask=ask,
                out=out,
            )
    except BaseException as exc:
        _record_failure(out, exc, attempts=recorder.completion_attempts)
        raise
    return out
