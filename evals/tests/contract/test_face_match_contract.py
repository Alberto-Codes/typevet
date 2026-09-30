"""Offline contract: an LFW face-match request through ScoringJudgmentAdapter.

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
from typevet_evals.datasets.lfw import LfwFace, LfwPair
from typevet_evals.face_match import (
    FACE_VISIBILITY,
    SAME_PERSON,
    VERDICT,
    VERDICT_LABELS,
    build_face_match_request,
    face_match_questions,
    judge_face_match,
)

pytestmark = pytest.mark.contract

MODEL = "fake-face-match"
PAIR = LfwPair(
    fold=3,
    left=LfwFace("Alpha_Example", 1),
    right=LfwFace("Bravo_Example", 2),
    same_person=False,
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
    return (ord(text[0]),) if text else ()


def _logs(probabilities: dict[str, float]) -> dict[str, float]:
    return {label: math.log(p) for label, p in probabilities.items()}


def _adapter() -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake(
        [
            _logs({"True": 0.3, "False": 0.7}),
            _logs({"same_person": 0.2, "different_person": 0.7, "cannot_tell": 0.1}),
            _logs({"0": 0.05, "1": 0.05, "2": 0.1, "3": 0.2, "4": 0.6}),
        ]
    )
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=_tokenize,
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    return adapter, fake


def test_questions_are_one_noul_one_choice_and_one_score() -> None:
    questions = face_match_questions()
    assert list(questions) == [SAME_PERSON, VERDICT, FACE_VISIBILITY]
    assert isinstance(questions[SAME_PERSON], Noul)
    verdict = questions[VERDICT]
    assert isinstance(verdict, Choice)
    assert tuple(verdict.criteria) == VERDICT_LABELS
    assert VERDICT_LABELS == ("same_person", "different_person", "cannot_tell")
    visibility = questions[FACE_VISIBILITY]
    assert isinstance(visibility, Score)
    assert len(visibility.criteria) == 5


def test_request_carries_both_images_in_pair_order() -> None:
    request = build_face_match_request(
        PAIR, left_image=RED, right_image=BLUE, mime_type="image/png"
    )
    assert request.media == (
        ImageInput(data=RED, mime_type="image/png"),
        ImageInput(data=BLUE, mime_type="image/png"),
    )
    assert request.pair == PAIR
    assert request.gold_same_person is False
    assert request.pair_id == "3:Alpha_Example_0001:Bravo_Example_0002"


def test_request_defaults_to_jpeg_media() -> None:
    request = build_face_match_request(PAIR, left_image=RED, right_image=BLUE)
    assert {image.mime_type for image in request.media} == {"image/jpeg"}


def test_fake_port_sees_two_images_in_order_and_returns_three_answers() -> None:
    adapter, fake = _adapter()
    request = build_face_match_request(
        PAIR, left_image=RED, right_image=BLUE, mime_type="image/png"
    )
    response = judge_face_match(adapter, request, MODEL)

    assert len(fake.calls) == 3
    for call in fake.calls:
        assert [image.data for image in call.media] == [RED, BLUE]
        assert count_media_markers(call.prefix) == 2
        assert "Alpha" not in call.prefix
        assert "Bravo" not in call.prefix
    assert response.nouls[SAME_PERSON].noul == pytest.approx(0.3)
    assert response.choices[VERDICT].choice == "different_person"
    visibility = response.scores[FACE_VISIBILITY]
    assert sorted(visibility.legend) == [0, 1, 2, 3, 4]
    assert set(response.answers) == {SAME_PERSON, VERDICT, FACE_VISIBILITY}


def test_state_names_image_two_for_the_visibility_score() -> None:
    request = build_face_match_request(PAIR, left_image=RED, right_image=BLUE)
    visibility = request.questions[FACE_VISIBILITY]
    assert isinstance(visibility, Score)
    assert "image 2" in str(visibility.instructions)
    assert "Alpha" not in request.state
    assert "Bravo" not in request.state
