"""Unit tests for the typevet judgevet bridge mapping (#284)."""

from __future__ import annotations

import asyncio
import multiprocessing
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest
from judgevet.domain.answers import ChoiceAnswer as JevChoiceAnswer
from judgevet.domain.answers import NoulAnswer as JevNoulAnswer
from judgevet.domain.answers import ScoreAnswer as JevScoreAnswer
from judgevet.domain.media import ImageAttachment, ImageEvidence, MediaCapabilities
from judgevet.domain.questions import Choice, Noul, Score
from judgevet.domain.usage import Usage
from judgevet.media import judge_with_images
from judgevet.providers import (
    ProviderCapabilityError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTransportError,
)

from tests.fixtures.judgevet_bridge import (
    FAKE_MODEL,
    LOGPROBS,
    QUESTIONS,
    STATE,
    judgment_port,
)
from tests.fixtures.judgevet_isolation import (
    bridge_import_error_without_judgevet,
    judgevet_modules_after_bare_import,
)
from typevet.adapters.inbound.judgevet import (
    DEFAULT_MAX_CHOICE_OPTIONS,
    AsyncTypevetSystemOnePort,
    BridgeCapabilities,
    TypevetMediaSystemOnePort,
    TypevetSystemOnePort,
)
from typevet.domain.decisions import MAX_ENUM_CHOICES
from typevet.domain.errors import (
    BackendHttpError,
    DecisionExecutionError,
    GemmaTemplateError,
    GenerationError,
    GenerationUnsupportedCapabilityError,
    JudgmentError,
    JudgmentValidationError,
    SchemaValidationError,
    ScoringError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul as TvNoul
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import ImageInput
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.unit

PNG = b"\x89PNG\r\n\x1a\nbridge"


class StubJudgment:
    """Record ``judge`` calls and return or raise a scripted outcome."""

    def __init__(
        self,
        *,
        response: JudgmentResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Store the scripted response or error."""
        self.response = response
        self.error = error
        self.calls: list[tuple[Any, Mapping[str, Any], str, Any]] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        self.calls.append((state, questions, model, media))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _noul_response(model: str = "resolved-gemma") -> JudgmentResponse:
    return JudgmentResponse(
        model=model,
        usage=TokenUsage(input_tokens=11, output_tokens=None),
        answers={"q": NoulAnswer(noul=0.7)},
    )


def test_typed_questions_map_to_typed_answers() -> None:
    port = TypevetSystemOnePort(judgment_port())
    response = port.system_one(STATE, QUESTIONS, FAKE_MODEL)
    noul = response.answers["conformance_noul"]
    choice = response.answers["conformance_choice"]
    score = response.answers["conformance_score"]
    assert isinstance(noul, JevNoulAnswer)
    assert noul.noul == pytest.approx(0.8)
    assert isinstance(choice, JevChoiceAnswer)
    assert choice.choice == "billing"
    assert choice.probabilities == pytest.approx({"billing": 0.75, "technical": 0.25})
    assert isinstance(score, JevScoreAnswer)
    assert score.legend == {0: "calm", 1: "concerned", 2: "angry"}
    assert score.score == pytest.approx(1.1)


def test_raw_wire_questions_map_like_typed_questions() -> None:
    raw = {
        "conformance_noul": {"type": "noul", "instructions": "Duplicate charge?"},
        "conformance_choice": {
            "type": "choice",
            "criteria": {"billing": "Money", "technical": "Bugs"},
        },
        "conformance_score": {"type": "score", "criteria": ["a", "b", "c"]},
    }
    response = TypevetSystemOnePort(judgment_port()).system_one(STATE, raw, FAKE_MODEL)
    assert set(response.answers) == set(raw)
    assert response.choices["conformance_choice"].choice == "billing"


def test_resolved_model_and_usage_come_from_typevet() -> None:
    stub = StubJudgment(response=_noul_response())
    response = TypevetSystemOnePort(stub).system_one("s", {"q": Noul()}, "requested")
    assert response.model == "resolved-gemma"
    assert response.usage == Usage(input_tokens=11, output_tokens=None)
    _, questions, model, media = stub.calls[0]
    assert model == "requested"
    assert media is None
    assert isinstance(questions["q"], TvNoul)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TransportError("refused"), ProviderTransportError),
        (
            BackendHttpError("HTTP 500", status_code=500, body_snippet=""),
            ProviderTransportError,
        ),
        (
            BackendHttpError("HTTP 400", status_code=400, body_snippet=""),
            ProviderRequestError,
        ),
        (GenerationUnsupportedCapabilityError("no images"), ProviderCapabilityError),
        (ScoringUnsupportedCapabilityError("no logprobs"), ProviderCapabilityError),
        (GemmaTemplateError("unknown template"), ProviderCapabilityError),
        (JudgmentValidationError("pinned model"), ProviderRequestError),
        (DecisionExecutionError("bad decision"), ProviderRequestError),
        (ScoringValidationError("missing score"), ProviderResponseError),
        (SchemaValidationError("bad shape"), ProviderResponseError),
        (GenerationError("missing top_logprobs"), ProviderResponseError),
        (JudgmentError("judgment failed"), ProviderResponseError),
        (ScoringError("scoring failed"), ProviderResponseError),
    ],
    ids=lambda value: type(value).__name__,
)
def test_typevet_errors_map_to_provider_errors(
    error: GenerationError, expected: type[ProviderError]
) -> None:
    port = TypevetSystemOnePort(StubJudgment(error=error))
    with pytest.raises(expected) as caught:
        port.system_one("s", {"q": Noul()}, "m")
    assert type(caught.value) is expected
    assert str(caught.value) == str(error)
    assert caught.value.__cause__ is error


def test_unexpected_errors_keep_their_type() -> None:
    error = RuntimeError("not a typevet failure")
    port = TypevetSystemOnePort(StubJudgment(error=error))
    with pytest.raises(RuntimeError) as caught:
        port.system_one("s", {"q": Noul()}, "m")
    assert caught.value is error


def test_capabilities_are_declared() -> None:
    port = TypevetSystemOnePort(judgment_port())
    assert DEFAULT_MAX_CHOICE_OPTIONS == MAX_ENUM_CHOICES == 24
    assert port.bridge_capabilities == BridgeCapabilities(
        logprobs_required=True, max_choice_options=24, media=None
    )


def _sized(kind: str, count: int) -> Choice | Score:
    if kind == "choice":
        return Choice(criteria={f"label{i}": None for i in range(count)})
    return Score(criteria=[f"level {i}" for i in range(count)])


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_options_above_the_cap_are_refused_before_judgment(kind: str) -> None:
    stub = StubJudgment(response=_noul_response())
    with pytest.raises(ProviderCapabilityError, match=r"has 25 options.*at most 24"):
        TypevetSystemOnePort(stub).system_one("s", {"q": _sized(kind, 25)}, "m")
    assert stub.calls == []


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_options_at_the_cap_reach_typevet(kind: str) -> None:
    stub = StubJudgment(response=_noul_response())
    TypevetSystemOnePort(stub).system_one("s", {"q": _sized(kind, 24)}, "m")
    ((_, questions, _, _),) = stub.calls
    assert len(questions["q"].criteria) == 24


def test_ten_options_reach_the_real_typevet_path() -> None:
    labels = {f"label{i}": None for i in range(10)}
    logprobs = {label: -1.0 for label in labels}
    judgment = judgment_port(scorer=ScriptedScoringFake(logprobs=logprobs))
    response = TypevetSystemOnePort(judgment).system_one(
        "s", {"q": Choice(criteria=labels)}, FAKE_MODEL
    )
    assert set(response.choices["q"].probabilities) == set(labels)


def test_a_lower_cap_is_honoured() -> None:
    stub = StubJudgment(response=_noul_response())
    port = TypevetSystemOnePort(stub, max_choice_options=2)
    assert port.bridge_capabilities.max_choice_options == 2
    with pytest.raises(ProviderCapabilityError, match="at most 2"):
        port.system_one(
            "s", {"q": Choice(criteria={"a": "A", "b": "B", "c": "C"})}, "m"
        )


def test_a_cap_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        TypevetSystemOnePort(StubJudgment(), max_choice_options=1)


def test_non_text_instructions_are_refused() -> None:
    stub = StubJudgment(response=_noul_response())
    with pytest.raises(ProviderCapabilityError, match="text instructions"):
        TypevetSystemOnePort(stub).system_one(
            "s", {"q": Noul(instructions={"rule": "x"})}, "m"
        )
    assert stub.calls == []


@pytest.mark.parametrize(
    "question",
    [
        {"type": "ranking"},
        {"type": "noul", "extra": 1},
        {"type": "choice", "criteria": ["a"]},
        {"type": "score", "criteria": "abc"},
        {"type": "noul", "criteria": ["a"]},
        42,
    ],
    ids=["type", "field", "choice", "score", "noul", "object"],
)
def test_malformed_questions_are_request_errors(question: Any) -> None:
    stub = StubJudgment(response=_noul_response())
    with pytest.raises(ProviderRequestError):
        TypevetSystemOnePort(stub).system_one("s", {"q": question}, "m")
    assert stub.calls == []


@pytest.mark.parametrize(
    "answers",
    [{}, {"q": NoulAnswer(noul=0.5), "extra": NoulAnswer(noul=0.5)}, {"q": 0.5}],
    ids=["missing", "extra", "untyped"],
)
def test_answers_that_break_the_contract_are_response_errors(answers: Any) -> None:
    response = JudgmentResponse(model="m", answers=answers)
    port = TypevetSystemOnePort(StubJudgment(response=response))
    with pytest.raises(ProviderResponseError):
        port.system_one("s", {"q": Noul()}, "m")


def _evidence(bindings: Mapping[str, list[str]]) -> ImageEvidence:
    images = [
        ImageAttachment("a", PNG, "image/png"),
        ImageAttachment("b", PNG + b"2", "image/png"),
    ]
    return ImageEvidence(images, bindings)


def test_media_port_sends_every_image_to_every_question() -> None:
    stub = StubJudgment(response=_noul_response())
    port = TypevetMediaSystemOnePort(stub, media=MediaCapabilities({"image/png"}))
    response = judge_with_images(
        port, "s", {"q": Noul()}, "m", evidence=_evidence({"q": ["a", "b"]})
    )
    assert response.model == "resolved-gemma"
    media = stub.calls[0][3]
    assert media == (
        ImageInput(data=PNG, mime_type="image/png"),
        ImageInput(data=PNG + b"2", mime_type="image/png"),
    )
    assert port.bridge_capabilities.media == MediaCapabilities({"image/png"})


class PerCallJudgment:
    """Answer each requested Noul question; script the model and usage per call."""

    def __init__(self, models: list[str], usages: list[TokenUsage]) -> None:
        """Store one model id and one usage per expected call."""
        self.models = models
        self.usages = usages
        self.calls: list[tuple[tuple[str, ...], tuple[ImageInput, ...] | None]] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        index = len(self.calls)
        self.calls.append((tuple(questions), media))
        answers: dict[str, Any] = {name: NoulAnswer(noul=0.5) for name in questions}
        return JudgmentResponse(
            model=self.models[index], usage=self.usages[index], answers=answers
        )


A = ImageInput(data=PNG, mime_type="image/png")
B = ImageInput(data=PNG + b"2", mime_type="image/png")
MIXED = {"q": Noul(), "r": Noul(), "s": Noul(), "t": Noul()}


def test_media_port_judges_each_bound_image_set_once() -> None:
    usages = [TokenUsage(3, 1), TokenUsage(5, 2), TokenUsage(7, None)]
    stub = PerCallJudgment(["g", "g", "g"], usages)
    port = TypevetMediaSystemOnePort(stub, media=MediaCapabilities({"image/png"}))
    bindings = {"q": ["a"], "s": ["b"], "t": ["b"]}
    response = port.system_one_media("x", MIXED, "m", evidence=_evidence(bindings))
    assert stub.calls == [(("q",), (A,)), (("r",), None), (("s", "t"), (B,))]
    assert list(response.answers) == ["q", "r", "s", "t"]
    assert response.model == "g"
    assert response.usage == Usage(input_tokens=15, output_tokens=None)


def test_media_port_keeps_image_order_per_group() -> None:
    stub = PerCallJudgment(["g", "g"], [TokenUsage(1, 1), TokenUsage(2, 2)])
    port = TypevetMediaSystemOnePort(stub, media=MediaCapabilities({"image/png"}))
    bindings = {"q": ["b", "a"], "r": ["a", "b"]}
    response = port.system_one_media(
        "x", {"q": Noul(), "r": Noul()}, "m", evidence=_evidence(bindings)
    )
    assert stub.calls == [(("q",), (B, A)), (("r",), (A, B))]
    assert response.usage == Usage(input_tokens=3, output_tokens=3)


def test_media_port_refuses_groups_that_disagree_on_the_model() -> None:
    stub = PerCallJudgment(["g", "h"], [TokenUsage(), TokenUsage()])
    port = TypevetMediaSystemOnePort(stub, media=MediaCapabilities({"image/png"}))
    with pytest.raises(ProviderResponseError, match="model"):
        port.system_one_media(
            "x",
            {"q": Noul(), "r": Noul()},
            "m",
            evidence=_evidence({"q": ["a"], "r": ["b"]}),
        )


def test_media_port_runs_mixed_bindings_on_the_real_typevet_path() -> None:
    scorer = ScriptedScoringFake(logprobs=LOGPROBS)
    judgment = judgment_port(media=True, scorer=scorer)
    port = TypevetMediaSystemOnePort(judgment, media=MediaCapabilities({"image/png"}))
    questions = {**QUESTIONS, "second_noul": Noul(instructions="Is it urgent?")}
    bindings = {
        "conformance_noul": ["a"],
        "conformance_score": ["b"],
        "second_noul": ["b"],
    }
    response = judge_with_images(
        port, STATE, questions, FAKE_MODEL, evidence=_evidence(bindings)
    )
    assert set(response.answers) == set(questions)
    assert {call.media for call in scorer.calls} == {(A,), (), (B,)}


def test_media_port_rejects_undeclarable_image_types() -> None:
    with pytest.raises(ValueError, match="does not support"):
        TypevetMediaSystemOnePort(
            StubJudgment(), media=MediaCapabilities({"image/png", "image/gif"})
        )


def test_media_port_refuses_unsupported_images_on_a_direct_call() -> None:
    stub = StubJudgment(response=_noul_response())
    port = TypevetMediaSystemOnePort(stub, media=MediaCapabilities({"image/png"}))
    evidence = ImageEvidence(
        [ImageAttachment("a", b"GIF89a", "image/gif")], {"q": ["a"]}
    )
    with pytest.raises(ProviderCapabilityError, match="supported set"):
        port.system_one_media("s", {"q": Noul()}, "m", evidence=evidence)
    assert stub.calls == []


def test_media_port_runs_the_real_typevet_media_path() -> None:
    scorer = ScriptedScoringFake(logprobs=LOGPROBS)
    judgment = judgment_port(media=True, scorer=scorer)
    port = TypevetMediaSystemOnePort(judgment, media=MediaCapabilities({"image/png"}))
    evidence = ImageEvidence(
        [ImageAttachment("a", PNG, "image/png")],
        {name: ["a"] for name in QUESTIONS},
    )
    response = judge_with_images(port, STATE, QUESTIONS, FAKE_MODEL, evidence=evidence)
    assert set(response.answers) == set(QUESTIONS)
    assert all(call.media == (ImageInput(PNG, "image/png"),) for call in scorer.calls)


def test_async_port_answers_through_the_sync_bridge() -> None:
    port = AsyncTypevetSystemOnePort(TypevetSystemOnePort(judgment_port()))
    response = asyncio.run(port.system_one(STATE, QUESTIONS, FAKE_MODEL))
    assert set(response.answers) == set(QUESTIONS)
    assert port.bridge_capabilities.logprobs_required is True


def _in_fresh_interpreter[T](probe: Callable[[], T]) -> T:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(probe).result(timeout=120)


def test_bare_typevet_imports_do_not_load_judgevet() -> None:
    assert _in_fresh_interpreter(judgevet_modules_after_bare_import) == []


def test_the_bridge_names_the_extra_when_judgevet_is_missing() -> None:
    message = _in_fresh_interpreter(bridge_import_error_without_judgevet)
    assert "typevet[judgevet]" in message
