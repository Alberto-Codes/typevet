"""Offline contract: a CEDAR signature request through ScoringJudgmentAdapter.

The fake scorer replaces the served model. These tests prove that one pair
becomes one judgment with both images in order and three typed answers. They
say nothing about model quality. Images are synthetic solid-colour PNGs.
"""

from __future__ import annotations

import math
import struct
import zlib

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import Choice, ImageInput, Noul, Score
from typevet.domain.media import count_media_markers
from typevet_evals.datasets.cedar import CedarPair, CedarSignature, PairKind
from typevet_evals.signature_match import (
    IMAGE_QUALITY,
    SAME_WRITER,
    VERDICT,
    VERDICT_LABELS,
    build_signature_match_request,
    judge_signature_match,
    signature_match_questions,
)

pytestmark = pytest.mark.contract

MODEL = "fake-signature-match"
PAIR = CedarPair(
    PairKind.GENUINE_SKILLED,
    CedarSignature(12, 3, forged=False),
    CedarSignature(12, 7, forged=True),
)


def _solid_png(rgb: tuple[int, int, int], size: int = 2) -> bytes:
    """Return a solid-colour RGB PNG of ``size`` by ``size`` pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


RED = _solid_png((255, 0, 0))
BLUE = _solid_png((0, 0, 255))


def _tokenize(text: str) -> tuple[int, ...]:
    return tuple(ord(char) for char in text)


def _logs(probabilities: dict[str, float]) -> dict[str, float]:
    return {label: math.log(p) for label, p in probabilities.items()}


def _adapter() -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake(
        [
            _logs({"True": 0.35, "False": 0.65}),
            _logs(
                {
                    "same_writer": 0.2,
                    "different_writer": 0.1,
                    "skilled_forgery_suspected": 0.6,
                    "cannot_tell": 0.1,
                }
            ),
            _logs({"0": 0.05, "1": 0.05, "2": 0.1, "3": 0.6, "4": 0.2}),
        ]
    )
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=_tokenize,
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    return adapter, fake


def test_questions_are_one_noul_one_choice_and_one_score() -> None:
    questions = signature_match_questions()
    assert list(questions) == [SAME_WRITER, VERDICT, IMAGE_QUALITY]
    assert isinstance(questions[SAME_WRITER], Noul)
    verdict = questions[VERDICT]
    assert isinstance(verdict, Choice)
    assert tuple(verdict.criteria) == VERDICT_LABELS
    assert VERDICT_LABELS == (
        "same_writer",
        "different_writer",
        "skilled_forgery_suspected",
        "cannot_tell",
    )
    quality = questions[IMAGE_QUALITY]
    assert isinstance(quality, Score)
    assert len(quality.criteria) == 5
    assert "image 2" in str(quality.instructions)


def test_request_carries_both_images_in_pair_order() -> None:
    request = build_signature_match_request(
        PAIR, reference_image=RED, questioned_image=BLUE
    )
    assert request.media == (
        ImageInput(data=RED, mime_type="image/png"),
        ImageInput(data=BLUE, mime_type="image/png"),
    )
    assert request.pair == PAIR
    assert request.pair_id == "genuine_skilled:original_12_3:forgeries_12_7"
    assert request.gold_kind is PairKind.GENUINE_SKILLED
    assert request.gold_same_writer is False


def test_fake_port_sees_two_images_in_order_and_returns_three_answers() -> None:
    adapter, fake = _adapter()
    request = build_signature_match_request(
        PAIR, reference_image=RED, questioned_image=BLUE
    )
    response = judge_signature_match(adapter, request, MODEL)

    assert len(fake.calls) == 3
    for call in fake.calls:
        assert [image.data for image in call.media] == [RED, BLUE]
        assert count_media_markers(call.prefix) == 2
        assert "forgeries_" not in call.prefix
        assert "original_" not in call.prefix
    assert response.nouls[SAME_WRITER].noul == pytest.approx(0.35)
    assert response.choices[VERDICT].choice == "skilled_forgery_suspected"
    quality = response.scores[IMAGE_QUALITY]
    assert sorted(quality.legend) == [0, 1, 2, 3, 4]
    assert set(response.answers) == {SAME_WRITER, VERDICT, IMAGE_QUALITY}


def test_state_and_questions_do_not_reveal_the_pair_kind() -> None:
    request = build_signature_match_request(
        PAIR, reference_image=RED, questioned_image=BLUE
    )
    texts = [request.state]
    texts += [str(question.instructions) for question in request.questions.values()]
    for text in texts:
        assert "12" not in text
        assert "genuine" not in text.lower()
        assert "original_" not in text
        assert "forgeries_" not in text
