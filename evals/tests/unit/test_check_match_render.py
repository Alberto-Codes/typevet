"""Unit tests for the synthetic check renderer and contact sheet (#315).

Byte determinism holds within one process and one Pillow and FreeType
build. A different Pillow build can change the rendered bytes.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from PIL import Image

from typevet_evals.check_match import (
    CheckVariant,
    check_cases,
    render_check,
    render_check_image,
    write_contact_sheet,
)

pytestmark = pytest.mark.unit

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
EVALS_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PAPER = (246, 243, 232)
# Area between the words line and the memo line: only SPECIMEN draws here.
SPECIMEN_REGION = (440, 250, 590, 320)
SIGNATURE_REGION = (600, 300, 940, 350)
MIN_MARK_PIXELS = 200
METADATA_CHUNKS = {b"tEXt", b"iTXt", b"zTXt", b"tIME"}


def _pixels(image: Image.Image, box: tuple[int, int, int, int]) -> list[bytes]:
    """Return the RGB pixels of ``box`` as 3-byte values."""
    data = image.crop(box).convert("RGB").tobytes()
    return [data[i : i + 3] for i in range(0, len(data), 3)]


def _png_chunk_types(data: bytes) -> list[bytes]:
    """Return the chunk types of a PNG file in order."""
    types: list[bytes] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        types.append(data[offset + 4 : offset + 8])
        offset += 12 + length
    return types


def _row_cases(row_index: int = 0) -> dict[CheckVariant, bytes]:
    return {
        case.variant: render_check(case)
        for case in check_cases()
        if case.row.index == row_index
    }


def test_same_seed_gives_same_png_bytes() -> None:
    first = [render_check(case) for case in check_cases(seed=0)[:14]]
    second = [render_check(case) for case in check_cases(seed=0)[:14]]
    assert first == second
    assert all(data.startswith(PNG_SIGNATURE) for data in first)


def test_each_variant_renders_a_different_image() -> None:
    renders = _row_cases()
    assert len(set(renders.values())) == len(CheckVariant)


def test_render_is_an_rgb_png_of_fixed_size() -> None:
    renders = _row_cases()
    sizes = set()
    for data in renders.values():
        with Image.open(io.BytesIO(data)) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            sizes.add(image.size)
    assert len(sizes) == 1


def test_low_legibility_render_is_the_clean_render_blurred() -> None:
    renders = _row_cases()
    with (
        Image.open(io.BytesIO(renders[CheckVariant.CLEAN])) as clean,
        Image.open(io.BytesIO(renders[CheckVariant.LOW_LEGIBILITY])) as blurred,
    ):
        clean_extent = clean.convert("L").getextrema()
        blurred_extent = blurred.convert("L").getextrema()
        assert isinstance(clean_extent[0], int)
        assert isinstance(blurred_extent[0], int)
        assert blurred_extent[0] > clean_extent[0]


def test_contact_sheet_writes_one_png_to_the_given_path(tmp_path: Path) -> None:
    cases = check_cases()[:7]
    target = tmp_path / "sheets" / "row_00.png"
    written = write_contact_sheet(cases, target, columns=7)
    assert written == target
    data = target.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    with Image.open(io.BytesIO(data)) as sheet:
        assert sheet.width > sheet.height


def test_contact_sheet_rejects_an_empty_case_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="case"):
        write_contact_sheet([], tmp_path / "empty.png")


def test_every_sharp_render_carries_the_specimen_and_void_marks() -> None:
    paper = bytes(PAPER)
    for case in check_cases()[:14]:
        if case.variant is CheckVariant.LOW_LEGIBILITY:
            continue
        image = render_check_image(case)
        specimen = [p for p in _pixels(image, SPECIMEN_REGION) if p != paper]
        void = [
            p
            for p in _pixels(image, SIGNATURE_REGION)
            if p[0] - p[1] > 40 and p[0] - p[2] > 40
        ]
        assert len(specimen) >= MIN_MARK_PIXELS, case.case_id
        assert len(void) >= MIN_MARK_PIXELS, case.case_id


def test_png_holds_no_text_or_time_chunk() -> None:
    data = render_check(check_cases()[0])
    types = _png_chunk_types(data)
    assert types[0] == b"IHDR"
    assert types[-1] == b"IEND"
    assert METADATA_CHUNKS.isdisjoint(types)


def test_no_image_file_under_evals_fixtures() -> None:
    images = [
        path
        for path in EVALS_FIXTURES.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    assert EVALS_FIXTURES.is_dir()
    assert images == []
