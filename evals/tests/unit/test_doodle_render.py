"""Unit tests for the 512 px doodle renderer (#412)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from typevet_evals.doodle_duel import RENDER_SIZE, render_strokes

pytestmark = pytest.mark.unit

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE = (((0, 128), (255, 128)),)


def _image(png: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png))
    image.load()
    return image


def test_render_is_512_png() -> None:
    image = _image(render_strokes(LINE))
    assert RENDER_SIZE == 512
    assert image.format == "PNG"
    assert image.size == (512, 512)
    assert image.mode == "RGB"


def test_render_draws_black_strokes_on_white() -> None:
    image = _image(render_strokes(LINE))
    assert image.getpixel((0, 0)) == WHITE
    assert image.getpixel((256, 100)) == WHITE
    assert image.getpixel((256, 257)) == BLACK
    assert image.getpixel((16, 257)) == BLACK
    assert image.getpixel((496, 257)) == BLACK
    assert image.getpixel((8, 257)) == WHITE


def test_render_is_deterministic() -> None:
    assert render_strokes(LINE) == render_strokes(LINE)
    assert render_strokes(LINE) != render_strokes((((0, 0), (255, 255)),))


def test_single_point_stroke_draws_a_dot() -> None:
    image = _image(render_strokes((((128, 128),),)))
    centre = round(16 + 128 * 480 / 255)
    assert image.getpixel((centre, centre)) == BLACK
    colors = {color: count for count, color in image.getcolors(1 << 16) or []}
    assert 0 < colors[BLACK] < 100


@pytest.mark.parametrize("point", [(256, 0), (0, -1)])
def test_render_rejects_coordinate_out_of_range(point: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="0 to 255"):
        render_strokes((((0, 0), point),))
