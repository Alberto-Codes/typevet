"""Contract tests: model framing and scoring transport substitute independently (#174).

Two fake framings (Gemma 4 turns and a non-Gemma marker set) cross two fake
transports (llama-style and vLLM-style payloads). Each pair runs one Noul, one
Choice and one Score through one ``ScoringJudgmentAdapter``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING, Any

import pytest

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_START_OF_TURN,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
)
from typevet.adapters.outbound.judgment_scoring import (
    ScoringJudgmentAdapter,
    judge_with_scoring,
)
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
)
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput

if TYPE_CHECKING:
    from typevet.ports import ModelFramingPort

_GEMMA_MARKERS = (
    GEMMA4_TURN_OPEN,
    GEMMA4_TURN_CLOSE,
    GEMMA3_START_OF_TURN,
    GEMMA3_END_OF_TURN,
)
_OTHER_OPEN = "[[USER]]"
_OTHER_CLOSE = "[[/USER]]"
_OTHER_ANSWER = "[[ASSISTANT]]"

_LOGPROBS = {
    "True": math.log(0.7),
    "False": math.log(0.3),
    "billing": math.log(0.8),
    "technical": math.log(0.2),
    "0": math.log(0.1),
    "1": math.log(0.3),
    "2": math.log(0.6),
}
_QUESTIONS = {
    "noul": Noul(instructions="Billing issue?", criteria={"true": "Yes"}),
    "route": Choice(
        criteria={"billing": "Money", "technical": "Bugs"},
        instructions="Pick:",
    ),
    "quality": Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:"),
}


class Gemma4Framing:
    """Fake framing that wraps the prefix in Gemma 4 turn markers."""

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return a Gemma 4 turn prefix ending with the no-thinking prefill.

        Returns:
            Prefix wrapped in ``<|turn>`` / ``<turn|>`` markers (#354).
        """
        del media
        return (
            f"{GEMMA4_TURN_OPEN}user\n{user_text}"
            f"{GEMMA4_TURN_CLOSE}\n{GEMMA4_TURN_OPEN}model\n"
            f"{GEMMA4_NO_THINKING_PREFILL}"
        )


class BracketFraming:
    """Fake non-Gemma framing with a distinct marker set."""

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return a bracket-marker prefix ending at the answer header.

        Returns:
            Prefix wrapped in the bracket marker set.
        """
        del media
        return f"{_OTHER_OPEN}{user_text}{_OTHER_CLOSE}{_OTHER_ANSWER}"


class _RecordingTransport:
    """Base fake transport: build a private payload, record it, score by label.

    Attributes:
        payload_keys (tuple[str, ...]): Keys of the private wire payload.
        sentinel (str): Distinctive transport-only value in each payload.
        payloads (list[dict[str, Any]]): Recorded private payloads.
        prefixes (list[str]): Recorded scoring prefixes.
    """

    payload_keys: tuple[str, ...] = ()
    sentinel: str = ""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.prefixes: list[str] = []

    def _payload(self, request: CandidateScoringRequest) -> dict[str, Any]:
        raise NotImplementedError

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Record the private payload and score each candidate by label.

        Returns:
            Scores from the shared logprob table, in request order.
        """
        payload = self._payload(request)
        assert set(payload) == set(self.payload_keys)
        self.payloads.append(payload)
        self.prefixes.append(request.prefix)
        return CandidateScoringResult(
            model=request.model,
            stage=request.stage,
            candidates=tuple(
                ScoredCandidate(spec.label, spec.token_ids, _LOGPROBS[spec.label])
                for spec in request.candidates
            ),
        )


class LlamaStyleTransport(_RecordingTransport):
    """Fake transport with a llama.cpp-shaped completion payload."""

    payload_keys = ("prompt", "n_probs", "n_predict", "id_slot")
    sentinel = "__wire_sentinel_llama__"

    def _payload(self, request: CandidateScoringRequest) -> dict[str, Any]:
        return {
            "prompt": request.prefix,
            "n_probs": len(request.candidates),
            "n_predict": 1,
            "id_slot": self.sentinel,
        }


class VllmStyleTransport(_RecordingTransport):
    """Fake transport with a vLLM-shaped chat payload."""

    payload_keys = ("messages", "logprob_token_ids", "max_tokens", "request_id")
    sentinel = "__wire_sentinel_vllm__"

    def _payload(self, request: CandidateScoringRequest) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": request.prefix}],
            "logprob_token_ids": [s.token_ids[0] for s in request.candidates],
            "max_tokens": 1,
            "request_id": self.sentinel,
        }


_FRAMINGS = (Gemma4Framing, BracketFraming)
_TRANSPORTS = (LlamaStyleTransport, VllmStyleTransport)
_PAIRS = list(itertools.product(_FRAMINGS, _TRANSPORTS))
_ALL_PAYLOAD_KEYS = frozenset(
    key for transport in _TRANSPORTS for key in transport.payload_keys
)


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),)


def _run(framing: ModelFramingPort, transport: _RecordingTransport) -> JudgmentResponse:
    adapter = ScoringJudgmentAdapter(
        transport, tokenize_content=_tokenize, framing=framing
    )
    return adapter.judge("Charged twice.", _QUESTIONS, "fake-model")


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        found = {str(k) for k in value}
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, (list, tuple)):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for item in value for s in _strings(item)]
    if value is None or type(value) in (bool, int, float):
        return []
    msg = f"leak check cannot inspect value of type {type(value).__name__}"
    raise TypeError(msg)


_PREFIX_WINDOW = 20


def _prefix_windows(prefixes: list[str]) -> set[str]:
    return {
        prefix[i : i + _PREFIX_WINDOW]
        for prefix in prefixes
        for i in range(len(prefix) - _PREFIX_WINDOW + 1)
    }


_PAIR_IDS = [f"{f.__name__}-{t.__name__}" for f, t in _PAIRS]


@pytest.mark.contract
@pytest.mark.parametrize(("framing_cls", "transport_cls"), _PAIRS, ids=_PAIR_IDS)
def test_pair_returns_equal_answers_without_leaks(
    framing_cls: type, transport_cls: type
) -> None:
    """Each framing x transport pair gives equal answers and leaks nothing."""
    reference = _run(Gemma4Framing(), LlamaStyleTransport())
    transport = transport_cls()
    response = _run(framing_cls(), transport)

    assert all(transport.sentinel in p.values() for p in transport.payloads)
    serialized = _strings(dataclasses.asdict(response))
    sentinels = {t.sentinel for t in _TRANSPORTS}
    leaked = [s for s in serialized for w in sentinels if w in s]
    assert leaked == []
    windows = _prefix_windows(transport.prefixes)
    assert windows
    assert [s for s in serialized for w in windows if w in s] == []

    assert response == reference
    assert response.nouls["noul"].noul == pytest.approx(0.7)
    assert response.choices["route"].choice == "billing"
    assert response.scores["quality"].score == pytest.approx(1.5)

    assert len(transport.payloads) == len(_QUESTIONS)
    if framing_cls is BracketFraming:
        for prefix in transport.prefixes:
            for marker in (*_GEMMA_MARKERS, CHATML_IM_START):
                assert marker not in prefix
            assert prefix.startswith(_OTHER_OPEN)
    else:
        assert all(p.startswith(GEMMA4_TURN_OPEN) for p in transport.prefixes)

    exposed = _keys(dataclasses.asdict(response)) | {
        f.name for f in dataclasses.fields(response)
    }
    assert exposed.isdisjoint(_ALL_PAYLOAD_KEYS)
    assert not any(hasattr(response, key) for key in _ALL_PAYLOAD_KEYS)


@pytest.mark.contract
def test_framing_that_drops_media_markers_fails_before_scoring_io() -> None:
    """A framing whose prefix loses media markers fails closed before IO."""

    class MarkerDroppingFraming:
        """Fake framing that drops the user text and its media markers."""

        def compose_prefix(
            self, *, user_text: str, media: tuple[ImageInput, ...]
        ) -> str:
            """Return a prefix without the user text.

            Returns:
                Prefix that holds no media marker.
            """
            del user_text, media
            return f"{_OTHER_OPEN}{_OTHER_ANSWER}"

    transport = LlamaStyleTransport()
    adapter = ScoringJudgmentAdapter(
        transport, tokenize_content=_tokenize, framing=MarkerDroppingFraming()
    )
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    with pytest.raises(JudgmentValidationError, match="media marker"):
        adapter.judge("state", _QUESTIONS, "fake-model", media=(image,))
    assert transport.payloads == []


@pytest.mark.contract
def test_framing_carries_media_markers_to_transport() -> None:
    """A framing that keeps context carries media markers to the transport."""
    transport = VllmStyleTransport()
    adapter = ScoringJudgmentAdapter(
        transport, tokenize_content=_tokenize, framing=BracketFraming()
    )
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    response = adapter.judge("state", _QUESTIONS, "fake-model", media=(image,))
    assert response.choices["route"].choice == "billing"
    assert all(p.startswith(_OTHER_OPEN) for p in transport.prefixes)


@pytest.mark.contract
def test_judge_with_scoring_forwards_non_gemma_framing() -> None:
    """The function path forwards ``framing`` and matches the adapter path."""
    reference = _run(BracketFraming(), LlamaStyleTransport())
    transport = LlamaStyleTransport()
    response = judge_with_scoring(
        "Charged twice.",
        _QUESTIONS,
        "fake-model",
        scoring_port=transport,
        tokenize_content=_tokenize,
        framing=BracketFraming(),
    )
    assert response == reference
    assert len(transport.prefixes) == len(_QUESTIONS)
    for prefix in transport.prefixes:
        for marker in (*_GEMMA_MARKERS, CHATML_IM_START):
            assert marker not in prefix
        assert prefix.startswith(_OTHER_OPEN)


@pytest.mark.contract
def test_judge_with_scoring_rejects_framing_with_served_template() -> None:
    """The function path surfaces the adapter error before scoring IO."""
    transport = LlamaStyleTransport()
    with pytest.raises(ValueError, match="not both"):
        judge_with_scoring(
            "state",
            _QUESTIONS,
            "fake-model",
            scoring_port=transport,
            tokenize_content=_tokenize,
            framing=BracketFraming(),
            served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
        )
    assert transport.payloads == []
