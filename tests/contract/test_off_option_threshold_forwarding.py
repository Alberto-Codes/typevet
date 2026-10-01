"""Contract: ``off_option_threshold`` reaches the answer through each wrapper (#368).

A caller typed to ``JudgmentPort`` passes the #353 guard. Through the
llama.cpp factory session, the key-masking wrapper and ``CalibratedJudgment``
the receipt on the answer holds the threshold and the flag. The judgevet
bridge forwards the keyword; its response carries no receipt.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_off_option_threshold_forwarding.py
    ```

See Also:
    - [typevet.ports.judgment][]: ``JudgmentPort`` protocol
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from judgevet.domain.media import ImageAttachment, ImageEvidence, MediaCapabilities
from judgevet.domain.questions import Noul as JevNoul

from tests.fixtures.judgevet_bridge import FAKE_MODEL, STATE, judgment_port
from tests.fixtures.judgment_contract import (
    ContractJudgmentFake,
    get_threshold_parity_fixture,
)
from tests.fixtures.scoring_contract import OFF_OPTION_N_VOCAB, get_off_option_fixture
from typevet.adapters.inbound import load_calibration_map
from typevet.adapters.inbound.backend_settings import _KeyMaskingJudgmentPort
from typevet.adapters.inbound.judgevet import (
    AsyncTypevetSystemOnePort,
    TypevetMediaSystemOnePort,
    TypevetSystemOnePort,
)
from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.vllm.request_ids import RequestIdJudgmentPort
from typevet.domain.errors import JudgmentError
from typevet.domain.judgment_questions import Noul, Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort
from typevet.runtime import ScoringJudgmentAdapter
from typevet.runtime.calibrated_judgment import CalibratedJudgment
from typevet.runtime.llama_cpp_gemma_vision import open_gemma_native_vision_judgment
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

_MODEL = "gemma-4-test"
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_PNG = b"\x89PNG\r\n\x1a\nthreshold"


def _router(request: httpx.Request) -> httpx.Response:
    """Serve the llama.cpp routes the factory and one scoring call need.

    Args:
        request: The request the factory or the scoring adapter sends.

    Returns:
        A canned response; ``/completion`` holds the off-option fixture.
    """
    path = request.url.path
    if path == "/props":
        body = {"modalities": {"vision": True, "audio": False}, "media_marker": "<m>"}
        return httpx.Response(200, json=body)
    if path == "/apply-template":
        return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
    if path == "/completion":
        top = get_off_option_fixture()["top_logprobs"]
        return httpx.Response(
            200, json={"completion_probabilities": [{"top_logprobs": top}]}
        )
    return httpx.Response(404)


def _control_tokens(text: str) -> tuple[int, ...]:
    """Map the Noul control strings to the fixture candidate ids.

    Args:
        text: ``"0"`` for False or ``"1"`` for True.

    Returns:
        Token id 202 (probability 0.2) or 101 (probability 0.5).
    """
    return {"0": (202,), "1": (101,)}[text]


@pytest.mark.parametrize(("threshold", "flag"), [(0.25, True), (0.35, False)])
def test_factory_session_port_flags_off_option_mass(
    threshold: float, flag: bool
) -> None:
    settings = LlamaSettings(base_url="http://offline-router", timeout=30.0)
    transport = httpx.MockTransport(_router)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            model=_MODEL,
            n_vocab=OFF_OPTION_N_VOCAB,
            http_client=client,
            tokenize_content=_control_tokens,
        ) as session,
    ):
        port: JudgmentPort = session.port
        response = port.judge(
            "Charged twice.",
            {"q": Noul(instructions="Billing issue?")},
            _MODEL,
            off_option_threshold=threshold,
        )
    receipt = response.off_option["q"]
    assert receipt.off_option_mass == pytest.approx(0.3, abs=1e-12)
    assert receipt.off_option_threshold == threshold
    assert receipt.off_option_flag is flag


class _RecordingPort:
    """``JudgmentPort`` that records each threshold and delegates the call.

    Attributes:
        thresholds (list[float | None]): Threshold of each ``judge`` call.
        model (str | None): Model id to report on each response, or None to
            keep the offline port's model id.
    """

    def __init__(self, *, model: str | None = None) -> None:
        self.thresholds: list[float | None] = []
        self.model = model
        self._inner = judgment_port(media=True)

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Record ``off_option_threshold``, then judge with the offline port.

        Args:
            state: Content under evaluation.
            questions: Named questions.
            model: Model id.
            media: Images for every question.
            off_option_threshold: Threshold to record and forward.

        Returns:
            The offline port response, with ``self.model`` as its model id
            when set.
        """
        self.thresholds.append(off_option_threshold)
        response = self._inner.judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )
        if self.model is None:
            return response
        return dataclasses.replace(response, model=self.model)


def test_key_masking_wrapper_forwards_threshold() -> None:
    inner = _RecordingPort()
    port: JudgmentPort = _KeyMaskingJudgmentPort(inner, ())
    response = port.judge(STATE, {"q": Noul()}, FAKE_MODEL, off_option_threshold=0.25)
    assert inner.thresholds == [0.25]
    assert response.off_option["q"].off_option_threshold == 0.25
    port.judge(STATE, {"q": Noul()}, FAKE_MODEL)
    assert inner.thresholds == [0.25, None]


def test_judgevet_bridge_forwards_threshold() -> None:
    inner = _RecordingPort()
    port = TypevetSystemOnePort(inner)
    port.system_one(STATE, {"q": JevNoul()}, FAKE_MODEL, off_option_threshold=0.25)
    port.system_one(STATE, {"q": JevNoul()}, FAKE_MODEL)
    assert inner.thresholds == [0.25, None]


def test_async_judgevet_bridge_forwards_threshold() -> None:
    inner = _RecordingPort()
    port = AsyncTypevetSystemOnePort(TypevetSystemOnePort(inner))
    asyncio.run(
        port.system_one(STATE, {"q": JevNoul()}, FAKE_MODEL, off_option_threshold=0.25)
    )
    assert inner.thresholds == [0.25]


def test_media_judgevet_bridge_forwards_threshold_to_every_group() -> None:
    inner = _RecordingPort()
    port = TypevetMediaSystemOnePort(inner, media=MediaCapabilities({"image/png"}))
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    questions = {"q": JevNoul(), "r": JevNoul()}
    port.system_one_media(
        STATE, questions, FAKE_MODEL, evidence=evidence, off_option_threshold=0.25
    )
    assert inner.thresholds == [0.25, 0.25]


def test_calibrated_judgment_forwards_threshold() -> None:
    platt = Path(__file__).resolve().parents[1] / "fixtures" / "calibration"
    path = platt / "platt_noul_map.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    maps = {"fraud": load_calibration_map(path, sha256=digest)}
    inner = _RecordingPort(model=maps["fraud"].fitted_on.model)
    port: JudgmentPort = CalibratedJudgment(
        inner, maps, task_id="fraud-message", backend="llama_cpp"
    )
    response = port.judge(STATE, {"q": Noul()}, FAKE_MODEL, off_option_threshold=0.25)
    port.judge(STATE, {"q": Noul()}, FAKE_MODEL)
    assert inner.thresholds == [0.25, None]
    assert response.off_option["q"].off_option_threshold == 0.25


def _scoring_adapter() -> ScoringJudgmentAdapter:
    """Build the real scoring adapter over a scripted Noul scorer.

    Returns:
        A ``ScoringJudgmentAdapter`` with no network IO.
    """
    fake = ScriptedScoringFake(logprobs={"True": math.log(0.6), "False": math.log(0.4)})
    return ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))


class _RecordingScoringAdapter(ScoringJudgmentAdapter):
    """Real scoring adapter that records each threshold it receives.

    Attributes:
        thresholds (list[float | None]): Threshold of each ``judge`` call.
    """

    def __init__(self) -> None:
        fake = ScriptedScoringFake(
            logprobs={"True": math.log(0.6), "False": math.log(0.4)}
        )
        super().__init__(fake, tokenize_content=lambda s: (ord(s[0]),))
        self.thresholds: list[float | None] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Record ``off_option_threshold``, then judge with the real adapter.

        Args:
            state: Content under evaluation.
            questions: Named questions.
            model: Model id.
            media: Images for every question.
            off_option_threshold: Threshold to record and forward.

        Returns:
            The real adapter response.
        """
        self.thresholds.append(off_option_threshold)
        return super().judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )


def test_request_id_wrapper_forwards_threshold() -> None:
    inner = _RecordingScoringAdapter()
    port: JudgmentPort = RequestIdJudgmentPort(inner)
    response = port.judge(STATE, {"q": Noul()}, FAKE_MODEL, off_option_threshold=0.25)
    port.judge(STATE, {"q": Noul()}, FAKE_MODEL)
    assert inner.thresholds == [0.25, None]
    assert response.off_option["q"].off_option_threshold == 0.25


def _outcome(port: JudgmentPort, threshold: float | None) -> str:
    """Judge the shared parity fixture and name the outcome.

    Args:
        port: Port under test.
        threshold: Threshold from the fixture, valid or not.

    Returns:
        ``"accept"``, or the class name of the raised ``JudgmentError``.
    """
    fixture = get_threshold_parity_fixture()
    try:
        port.judge(
            fixture["state"],
            fixture["questions"],
            fixture["model"],
            off_option_threshold=threshold,
        )
    except JudgmentError as exc:
        return type(exc).__name__
    return "accept"


_PARITY = get_threshold_parity_fixture()


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(value, "JudgmentValidationError") for value in _PARITY["invalid"]]
    + [(value, "accept") for value in _PARITY["valid"]],
    ids=repr,
)
def test_fake_and_scoring_adapter_agree_on_threshold(
    threshold: float | None, expected: str
) -> None:
    fake = _outcome(ContractJudgmentFake(), threshold)
    real = _outcome(_scoring_adapter(), threshold)
    assert (fake, real) == (expected, expected)
