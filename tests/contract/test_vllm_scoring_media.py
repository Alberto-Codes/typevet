"""Contract tests for vLLM media content blocks (#107B).

Fixtures are redacted vLLM v0.30.0 responses from the #168 probe:
``image_three_way`` (P4) and ``image_binary`` (P3). The probe request held a
placeholder for the image bytes; these tests send synthetic PNGs instead.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.synthetic_images import solid_image
from typevet.adapters.outbound.vllm.scoring import (
    ChatContentFraming,
    VllmCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.judgment_questions import Choice
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_MODEL = "gemma-4-31b-it"
_CONTROL_IDS = {"0": 236771, "1": 236770, "2": 236778}
_P4_IDS = (236771, 236770, 236778, 250000, 30809)
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


def _logprob_by_id(name: str) -> dict[int, float]:
    content = _fixture(name)["response"]["choices"][0]["logprobs"]["content"]
    return {
        int(item["token"].removeprefix("token_id:")): item["logprob"]
        for item in content[0]["top_logprobs"]
    }


def _probe_text(name: str) -> str:
    blocks = _fixture(name)["request"]["messages"][0]["content"]
    return next(block["text"] for block in blocks if block["type"] == "text")


def _image_block(image: ImageInput) -> dict[str, Any]:
    encoded = base64.b64encode(image.data).decode("ascii")
    url = f"data:{image.mime_type};base64,{encoded}"
    return {"type": "image_url", "image_url": {"url": url}}


class _Recorder:
    """MockTransport handler that records bodies and replies with one payload."""

    def __init__(self, payload: object) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._payload = payload

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=self._payload)


def _adapter(recorder: _Recorder) -> VllmCandidateScoringAdapter:
    client = httpx.Client(
        transport=httpx.MockTransport(recorder), base_url="http://test"
    )
    return VllmCandidateScoringAdapter(base_url="http://test", client=client)


def _content(prefix: str, media: tuple[ImageInput, ...]) -> list[dict[str, Any]]:
    recorder = _Recorder(_fixture("image_three_way")["response"])
    request = CandidateScoringRequest(
        model=_MODEL,
        prefix=prefix,
        candidates=(CandidateTokenSpec("zero", (236771,)),),
        media=media,
    )
    _adapter(recorder).score_candidates(request)
    return recorder.bodies[0]["messages"][0]["content"]


def _texts(content: list[dict[str, Any]]) -> list[str]:
    return [block["text"] for block in content if block["type"] == "text"]


@pytest.mark.contract
def test_vllm_media_one_image_over_p4_scores_all_ids() -> None:
    image = solid_image("red")
    text = _probe_text("image_three_way")
    recorder = _Recorder(_fixture("image_three_way")["response"])
    request = CandidateScoringRequest(
        model=_MODEL,
        prefix=f"{MEDIA_MARKER}\n{text}",
        candidates=tuple(CandidateTokenSpec(f"id{i}", (i,)) for i in _P4_IDS),
        media=(image,),
    )

    result = _adapter(recorder).score_candidates(request)

    body = recorder.bodies[0]
    assert set(body) == _EXPECTED_KEYS
    assert body["logprob_token_ids"] == list(_P4_IDS)
    assert body["messages"] == [
        {
            "role": "user",
            "content": [_image_block(image), {"type": "text", "text": text}],
        }
    ]
    by_id = _logprob_by_id("image_three_way")
    assert [c.label for c in result.candidates] == [f"id{i}" for i in _P4_IDS]
    assert [c.logprob for c in result.candidates] == [by_id[i] for i in _P4_IDS]
    assert result.usage.input_tokens == 349
    assert result.usage.output_tokens == 1


@pytest.mark.contract
def test_vllm_media_two_images_keep_media_order() -> None:
    red, blue = solid_image("red"), solid_image("blue")
    content = _content(f"{MEDIA_MARKER}\n{MEDIA_MARKER}\nWhich colour?", (red, blue))
    assert content == [
        _image_block(red),
        _image_block(blue),
        {"type": "text", "text": "Which colour?"},
    ]
    reversed_content = _content(
        f"{MEDIA_MARKER}\n{MEDIA_MARKER}\nWhich colour?", (blue, red)
    )
    assert reversed_content[:2] == [_image_block(blue), _image_block(red)]


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (f"Before {MEDIA_MARKER}\nafter", ["Before ", "after"]),
        (f"{MEDIA_MARKER}after", ["after"]),
        (f"{MEDIA_MARKER}\n\nafter", ["\nafter"]),
        (f"a\n{MEDIA_MARKER}", ["a\n"]),
    ],
    ids=["text_around", "no_newline", "one_newline_only", "trailing_marker"],
)
@pytest.mark.contract
def test_vllm_media_text_blocks_split_at_markers(
    prefix: str, expected: list[str]
) -> None:
    content = _content(prefix, (solid_image("green"),))
    assert _texts(content) == expected
    assert sum(block["type"] == "image_url" for block in content) == 1


@pytest.mark.contract
def test_vllm_media_text_blocks_equal_framing_prefix_minus_markers() -> None:
    media = (solid_image("red"), solid_image("green"))
    prefix = ChatContentFraming().compose_prefix(
        context=f"{MEDIA_MARKER}\n{MEDIA_MARKER}\nExpense claim: total 12.",
        field_block="verdict: Pick one.",
        media=media,
    )
    content = _content(prefix, media)
    assert "".join(_texts(content)) == prefix.replace(f"{MEDIA_MARKER}\n", "")
    assert [block["type"] for block in content] == ["image_url", "image_url", "text"]


@pytest.mark.contract
@pytest.mark.parametrize(
    ("fixture", "criteria", "expected"),
    [
        ("image_three_way", ("supported", "contradicted", "insufficient"), 3),
        ("image_binary", ("supported", "contradicted"), 2),
    ],
    ids=["p4_three_way", "p3_binary"],
)
def test_vllm_media_judgment_choice_equals_scripted_fake(
    fixture: str, criteria: tuple[str, ...], expected: int
) -> None:
    by_id = _logprob_by_id(fixture)
    questions = {
        "verdict": Choice(
            criteria={name: f"Receipt: {name}" for name in criteria},
            instructions="Pick one:",
        )
    }
    scripted = {name: by_id[_CONTROL_IDS[str(i)]] for i, name in enumerate(criteria)}

    def tokenize(text: str) -> tuple[int, ...]:
        return (_CONTROL_IDS[text],)

    recorder = _Recorder(_fixture(fixture)["response"])
    vllm_port = ScoringJudgmentAdapter(
        _adapter(recorder), tokenize_content=tokenize, framing=ChatContentFraming()
    )
    fake = ScriptedScoringFake(logprobs=scripted)
    fake_port = ScoringJudgmentAdapter(
        fake, tokenize_content=tokenize, framing=ChatContentFraming()
    )
    media = (solid_image("blue"),)
    state = "Expense claim for this receipt: total 2,352,460."

    vllm_answer = vllm_port.judge(state, questions, _MODEL, media=media)
    fake_answer = fake_port.judge(state, questions, _MODEL, media=media)

    assert vllm_answer.choices == fake_answer.choices
    assert vllm_answer.choices["verdict"].choice == "supported"
    body = recorder.bodies[0]
    assert len(body["logprob_token_ids"]) == expected
    content = body["messages"][0]["content"]
    assert content[0] == _image_block(media[0])
    assert _texts(content) == [fake.calls[0].prefix.replace(f"{MEDIA_MARKER}\n", "")]
