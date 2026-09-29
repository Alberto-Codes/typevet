"""Contract tests for the vLLM chat-completions candidate scoring adapter (#107A).

Fixtures are redacted vLLM v0.30.0 responses from the #168 probe:
``text_three_way`` (P2), ``text_string_tokens`` (P1) and
``top20_without_ids`` (P12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters import outbound
from typevet.adapters.outbound.vllm_scoring import (
    ChatContentFraming,
    VllmCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import (
    GenerationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.scoring_stage import ScoreStage
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_MODEL = "gemma-4-31b-it"
_CONTROL_IDS = {"0": 236771, "1": 236770, "2": 236778}
_EXPECTED_KEYS = {
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "logprobs",
    "logprob_token_ids",
    "return_tokens_as_token_ids",
    "add_generation_prompt",
    "chat_template_kwargs",
}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _response_json(name: str) -> dict[str, Any]:
    return _fixture(name)["response"]


def _p2_logprob_by_id() -> dict[int, float]:
    content = _response_json("text_three_way")["choices"][0]["logprobs"]["content"]
    return {
        int(item["token"].removeprefix("token_id:")): item["logprob"]
        for item in content[0]["top_logprobs"]
    }


def _request(
    *,
    stage: ScoreStage = ScoreStage.PRE_SAMPLING,
    candidates: tuple[CandidateTokenSpec, ...] | None = None,
    prefix: str = "Answer:",
    media: tuple[ImageInput, ...] = (),
) -> CandidateScoringRequest:
    specs = candidates or (
        CandidateTokenSpec("zero", (236771,)),
        CandidateTokenSpec("one", (236770,)),
        CandidateTokenSpec("two", (236778,)),
    )
    return CandidateScoringRequest(
        model=_MODEL,
        prefix=prefix,
        candidates=specs,
        stage=stage,
        media=media,
    )


class _Recorder:
    """MockTransport handler that records bodies and replies with one payload."""

    def __init__(self, *, payload: object = None, content: bytes | None = None):
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self._payload = payload
        self._content = content

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.bodies.append(json.loads(request.content.decode()))
        if self._content is not None:
            return httpx.Response(
                200,
                content=self._content,
                headers={"content-type": "application/json"},
            )
        return httpx.Response(200, json=self._payload)


def _adapter(recorder: _Recorder) -> VllmCandidateScoringAdapter:
    client = httpx.Client(
        transport=httpx.MockTransport(recorder), base_url="http://test"
    )
    return VllmCandidateScoringAdapter(base_url="http://test", client=client)


@pytest.mark.contract
def test_vllm_scoring_body_shape_and_p2_scores() -> None:
    recorder = _Recorder(payload=_response_json("text_three_way"))
    result = _adapter(recorder).score_candidates(_request())

    assert recorder.paths == ["/v1/chat/completions"]
    body = recorder.bodies[0]
    assert set(body) == _EXPECTED_KEYS
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Answer:"}]
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["logprob_token_ids"] == [236771, 236770, 236778]
    assert body["return_tokens_as_token_ids"] is True
    assert body["add_generation_prompt"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "continue_final_message" not in body
    assert "top_logprobs" not in body

    by_id = _p2_logprob_by_id()
    assert [c.label for c in result.candidates] == ["zero", "one", "two"]
    assert [c.logprob for c in result.candidates] == [
        by_id[236771],
        by_id[236770],
        by_id[236778],
    ]
    assert result.model == _MODEL
    assert result.usage.input_tokens == 83
    assert result.usage.output_tokens == 1


@pytest.mark.contract
def test_vllm_scoring_logprob_token_ids_follow_request_order() -> None:
    recorder = _Recorder(payload=_response_json("text_three_way"))
    specs = (
        CandidateTokenSpec("two", (236778,)),
        CandidateTokenSpec("zero", (236771,)),
    )
    result = _adapter(recorder).score_candidates(_request(candidates=specs))
    assert recorder.bodies[0]["logprob_token_ids"] == [236778, 236771]
    assert [c.label for c in result.candidates] == ["two", "zero"]


@pytest.mark.contract
def test_chat_content_framing_has_no_turn_markers_and_keeps_media() -> None:
    framing = ChatContentFraming()
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    text = framing.compose_prefix(context="ctx", field_block="fields", media=())
    assert text == "ctx\n\nfields"
    context = f"{MEDIA_MARKER}\nctx"
    with_media = framing.compose_prefix(
        context=context, field_block="fields", media=(image,)
    )
    assert with_media == f"{MEDIA_MARKER}\nctx\n\nfields"


@pytest.mark.contract
def test_vllm_judgment_over_p2_equals_scripted_fake() -> None:
    by_id = _p2_logprob_by_id()
    questions = {
        "support": Noul(instructions="Does the receipt support the claim?"),
        "verdict": Choice(
            criteria={
                "supported": "Receipt supports the claim",
                "contradicted": "Receipt contradicts the claim",
                "insufficient": "Receipt is not enough",
            },
            instructions="Pick one:",
        ),
    }
    scripted = {
        "False": by_id[_CONTROL_IDS["0"]],
        "True": by_id[_CONTROL_IDS["1"]],
        "supported": by_id[_CONTROL_IDS["0"]],
        "contradicted": by_id[_CONTROL_IDS["1"]],
        "insufficient": by_id[_CONTROL_IDS["2"]],
    }

    def tokenize(text: str) -> tuple[int, ...]:
        return (_CONTROL_IDS[text],)

    recorder = _Recorder(payload=_response_json("text_three_way"))
    vllm_port = ScoringJudgmentAdapter(
        _adapter(recorder),
        tokenize_content=tokenize,
        framing=ChatContentFraming(),
    )
    fake = ScriptedScoringFake(logprobs=scripted)
    fake_port = ScoringJudgmentAdapter(
        fake, tokenize_content=tokenize, framing=ChatContentFraming()
    )
    state = "Expense claim for this receipt: total 2,352,460."

    vllm_answer = vllm_port.judge(state, questions, _MODEL)
    fake_answer = fake_port.judge(state, questions, _MODEL)

    assert vllm_answer.nouls == fake_answer.nouls
    assert vllm_answer.choices == fake_answer.choices
    assert vllm_answer.choices["verdict"].choice == "insufficient"
    sent = [body["messages"][0]["content"] for body in recorder.bodies]
    assert sent == [call.prefix for call in fake.calls]
    assert all(s.startswith(f"{state}\n\n") for s in sent)
    assert all("<|turn>" not in s and "<start_of_turn>" not in s for s in sent)


@pytest.mark.contract
def test_vllm_scoring_rejects_string_tokens_from_p1() -> None:
    recorder = _Recorder(payload=_response_json("text_string_tokens"))
    with pytest.raises(GenerationError, match="token_id"):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
def test_vllm_scoring_rejects_top20_without_requested_ids_p12() -> None:
    recorder = _Recorder(payload=_response_json("top20_without_ids"))
    with pytest.raises(ScoringValidationError, match="missing scores"):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
def test_vllm_scoring_rejects_negative_infinity() -> None:
    payload = _response_json("text_three_way")
    entries = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    for entry in entries:
        if entry["token"].endswith(":236770"):
            entry["logprob"] = float("-inf")
    raw = json.dumps(payload, allow_nan=True).encode()
    assert b"-Infinity" in raw
    recorder = _Recorder(content=raw)
    with pytest.raises(ScoringValidationError, match="non-finite"):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"stage": ScoreStage.POST_SAMPLING},
        {"candidates": (CandidateTokenSpec("pair", (1, 2)),)},
    ],
    ids=["post_sampling", "multi_token"],
)
def test_vllm_scoring_refuses_unsupported_capability_before_post(
    request_kwargs: dict[str, Any],
) -> None:
    recorder = _Recorder(payload=_response_json("text_three_way"))
    with pytest.raises(ScoringUnsupportedCapabilityError):
        _adapter(recorder).score_candidates(_request(**request_kwargs))
    assert recorder.bodies == []


@pytest.mark.contract
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"choices": []},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": {"content": []}}]},
        {"choices": [{"logprobs": {"content": [{"top_logprobs": "x"}]}}]},
        {"choices": [{"logprobs": {"content": [{"top_logprobs": ["x"]}]}}]},
        {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"x": 1}]}]}}]},
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"top_logprobs": [{"token": "token_id:x", "logprob": 0}]}
                        ]
                    }
                }
            ]
        },
    ],
    ids=[
        "root_list",
        "no_choices",
        "null_logprobs",
        "empty_content",
        "top_not_list",
        "entry_not_object",
        "entry_missing_fields",
        "token_id_not_int",
    ],
)
def test_vllm_scoring_bad_shape_raises_generation_error(payload: object) -> None:
    with pytest.raises(GenerationError):
        _adapter(_Recorder(payload=payload)).score_candidates(_request())


@pytest.mark.contract
def test_vllm_scoring_invalid_json_raises_generation_error() -> None:
    recorder = _Recorder(content=b"not json")
    with pytest.raises(GenerationError):
        _adapter(recorder).score_candidates(_request())


@pytest.mark.contract
def test_vllm_scoring_unknown_usage_stays_none() -> None:
    payload = _response_json("text_three_way")
    payload["usage"] = {"prompt_tokens": -1, "completion_tokens": True}
    result = _adapter(_Recorder(payload=payload)).score_candidates(_request())
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


@pytest.mark.contract
def test_vllm_scoring_owned_client_closes_on_exit() -> None:
    with VllmCandidateScoringAdapter("http://127.0.0.1:1") as adapter:
        client = adapter._ensure_client()
        assert not client.is_closed
    assert client.is_closed


@pytest.mark.contract
def test_vllm_scoring_adapter_is_exported_from_outbound_package() -> None:
    assert outbound.VllmCandidateScoringAdapter is VllmCandidateScoringAdapter
    assert "VllmCandidateScoringAdapter" in outbound.__all__
