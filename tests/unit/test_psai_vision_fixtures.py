"""Fixture integrity for the vendored PSAI vision smoke (#154)."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from typevet.domain.errors import ScoringValidationError
from typevet.evaluation.datasets.psai_vision import (
    ACCEPTED_UNIQUE_DATA_IDS,
    FOX_FAMILY,
    GOLD_FIELDS,
    MANIFEST_VERSION,
    MANUAL_VISUAL_PROVENANCE,
    NON_FOX_FAMILY,
    VISUAL_QUESTION_NAME,
    example_image_input,
    load_vision_smoke,
    vision_smoke_manifest_path,
)

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _manifest_text() -> str:
    return (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")


def _raw_manifest() -> dict[str, Any]:
    return json.loads(_manifest_text())


def _read_image(file_name: str) -> bytes:
    return (FIXTURE_DIR / file_name).read_bytes()


def test_vision_smoke_manifest_path_points_at_the_vendored_fixture() -> None:
    assert vision_smoke_manifest_path(FIXTURE_DIR) == FIXTURE_DIR / "manifest.json"


def test_manifest_holds_exactly_the_accepted_ids() -> None:
    fixture = load_vision_smoke(_manifest_text())
    ids = tuple(example.unique_data_id for example in fixture.examples)
    assert ids == ACCEPTED_UNIQUE_DATA_IDS


def test_manifest_preserves_mit_provenance() -> None:
    fixture = load_vision_smoke(_manifest_text())
    assert fixture.license == "MIT"
    assert fixture.dataset_id == "anaisleila/computer-use-data-psai"
    assert fixture.split == "train"
    assert fixture.screenshot_index == 0
    for example in fixture.examples:
        provenance = example.provenance(fixture)
        assert provenance["license"] == "MIT"
        assert provenance["dataset_id"] == fixture.dataset_id
        assert provenance["unique_data_id"] == example.unique_data_id
        assert provenance["screenshot_index"] == 0


def test_every_screenshot_file_matches_its_recorded_digest() -> None:
    fixture = load_vision_smoke(_manifest_text())
    for example in fixture.examples:
        shot = example.screenshot
        data = _read_image(shot.file_name)
        assert data[:8] == PNG_MAGIC
        assert len(data) == shot.byte_count
        assert hashlib.sha256(data).hexdigest() == shot.sha256
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (shot.width, shot.height)
        assert shot.mime_type == "image/png"


def test_fixture_holds_at_least_two_rows_per_visual_family() -> None:
    fixture = load_vision_smoke(_manifest_text())
    assert len(fixture.by_family(FOX_FAMILY)) >= 2
    assert len(fixture.by_family(NON_FOX_FAMILY)) >= 2


def test_gold_lives_only_under_expected() -> None:
    raw = _raw_manifest()
    for row in raw["rows"]:
        assert set(row["expected"]) == set(GOLD_FIELDS)
        outside = set(row) - {"expected"}
        assert not outside.intersection(GOLD_FIELDS)


def test_visual_question_is_labelled_manual_not_annotation_backed() -> None:
    fixture = load_vision_smoke(_manifest_text())
    assert fixture.gold_provenance[VISUAL_QUESTION_NAME] == MANUAL_VISUAL_PROVENANCE
    assert fixture.gold_provenance["category"] == "annotation"
    assert fixture.gold_provenance["requires_login"] == "annotation"


def test_expected_answers_match_the_visual_family() -> None:
    fixture = load_vision_smoke(_manifest_text())
    for example in fixture.examples:
        expected = example.expected()
        is_fox = example.visual_family == FOX_FAMILY
        assert expected[VISUAL_QUESTION_NAME] is is_fox
        assert example.is_fox is is_fox
        assert expected["category"] == "BROWSER_TASK"
        assert expected["requires_login"] is False


def test_example_image_input_carries_the_verified_png_bytes() -> None:
    fixture = load_vision_smoke(_manifest_text())
    example = fixture.examples[0]
    image = example_image_input(example, _read_image)
    assert image.mime_type == "image/png"
    assert hashlib.sha256(image.data).hexdigest() == example.screenshot.sha256


def test_example_image_input_rejects_bytes_that_miss_the_digest() -> None:
    fixture = load_vision_smoke(_manifest_text())
    example = fixture.examples[0]
    with pytest.raises(ScoringValidationError, match="sha256"):
        example_image_input(example, lambda _name: PNG_MAGIC + b"tampered")


def _mutated(change: Any) -> str:
    raw = _raw_manifest()
    change(raw)
    return json.dumps(raw)


@pytest.mark.parametrize(
    ("name", "change", "match"),
    [
        (
            "wrong_version",
            lambda raw: raw.__setitem__("manifest_version", MANIFEST_VERSION + 1),
            "manifest_version",
        ),
        (
            "missing_key",
            lambda raw: raw.pop("license"),
            "missing",
        ),
        (
            "wrong_license",
            lambda raw: raw.__setitem__("license", "CC-BY-4.0"),
            "license",
        ),
        (
            "unknown_family",
            lambda raw: raw["rows"][0].__setitem__("visual_family", "bbc_news"),
            "visual_family",
        ),
        (
            "expected_key_mismatch",
            lambda raw: raw["rows"][0]["expected"].__setitem__("difficulty", "EASY"),
            "expected",
        ),
        (
            "gold_outside_expected",
            lambda raw: raw["rows"][0].__setitem__("requires_login", False),
            "outside",
        ),
        (
            "duplicate_id",
            lambda raw: raw["rows"][1].__setitem__(
                "unique_data_id", raw["rows"][0]["unique_data_id"]
            ),
            "duplicate",
        ),
        (
            "one_family_only",
            lambda raw: raw.__setitem__("rows", raw["rows"][:3]),
            "at least",
        ),
        (
            "family_gold_disagreement",
            lambda raw: raw["rows"][0]["expected"].__setitem__(
                VISUAL_QUESTION_NAME, False
            ),
            "visual_family",
        ),
        (
            "unknown_gold_provenance",
            lambda raw: raw["gold_provenance"].__setitem__("category", "guessed"),
            "gold_provenance",
        ),
    ],
)
def test_load_vision_smoke_rejects_broken_manifests(
    name: str, change: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_vision_smoke(_mutated(change))


@pytest.mark.parametrize(
    ("name", "change", "match"),
    [
        (
            "wrong_dataset",
            lambda raw: raw.__setitem__("dataset_id", "someone/else"),
            "dataset_id",
        ),
        (
            "rows_not_array",
            lambda raw: raw.__setitem__("rows", {}),
            "array",
        ),
        (
            "row_not_object",
            lambda raw: raw["rows"].__setitem__(0, "nope"),
            "rows must be objects",
        ),
        (
            "screenshot_not_object",
            lambda raw: raw["rows"][0].__setitem__("screenshot", "nope"),
            "screenshot must be an object",
        ),
        (
            "expected_not_object",
            lambda raw: raw["rows"][0].__setitem__("expected", "nope"),
            "expected must be an object",
        ),
        (
            "gold_provenance_short",
            lambda raw: raw["gold_provenance"].pop("category"),
            "must name exactly",
        ),
        (
            "visual_marked_annotation",
            lambda raw: raw["gold_provenance"].__setitem__(
                VISUAL_QUESTION_NAME, "annotation"
            ),
            MANUAL_VISUAL_PROVENANCE,
        ),
    ],
)
def test_load_vision_smoke_rejects_broken_manifest_shapes(
    name: str, change: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_vision_smoke(_mutated(change))


def test_load_vision_smoke_rejects_non_object_manifest() -> None:
    with pytest.raises(ValueError, match="object"):
        load_vision_smoke("[]")


def test_load_vision_smoke_rejects_text_that_is_not_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        load_vision_smoke("{not json")


def test_by_family_rejects_an_unknown_family() -> None:
    fixture = load_vision_smoke(_manifest_text())
    with pytest.raises(ValueError, match="unknown visual_family"):
        fixture.by_family("bbc_news")
