"""Deterministic 512 px PNG renders of Quick, Draw! strokes (#412).

``render_strokes`` draws black lines of width 6 with curved joints on a
white 512 by 512 RGB canvas. Coordinates 0 to 255 map to pixels 16 to 496,
so a margin of 16 pixels stays white. A one-point stroke draws a dot. The
repository stores no render.

Attributes:
    RENDER_SIZE (int): Width and height of one render, in pixels.
    LINE_WIDTH (int): Stroke width, in pixels.
    MARGIN (int): White margin on each side, in pixels.

Examples:
    ```python
    from typevet_evals.doodle_duel.render import render_strokes

    png = render_strokes((((0, 0), (255, 255)),))
    assert png[1:4] == b"PNG"
    ```

See Also:
    - [typevet_evals.datasets.quickdraw][]: the strokes that this module draws
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Final

from PIL import Image, ImageDraw

RENDER_SIZE: Final[int] = 512
LINE_WIDTH: Final[int] = 6
MARGIN: Final[int] = 16

_MAX_COORDINATE: Final[int] = 255
_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)
_BLACK: Final[tuple[int, int, int]] = (0, 0, 0)
_SCALE: Final[float] = (RENDER_SIZE - 2 * MARGIN) / _MAX_COORDINATE


def _pixel(value: int) -> float:
    if not 0 <= value <= _MAX_COORDINATE:
        msg = f"stroke coordinate {value} is outside 0 to 255"
        raise ValueError(msg)
    return MARGIN + value * _SCALE


def render_strokes(strokes: Sequence[Sequence[tuple[int, int]]]) -> bytes:
    """Render strokes to PNG bytes.

    Args:
        strokes: Strokes, each a sequence of ``(x, y)`` points in 0 to 255.

    Returns:
        PNG bytes of a 512 by 512 RGB image.

    Raises:
        ValueError: When a coordinate is outside 0 to 255.
    """
    image = Image.new("RGB", (RENDER_SIZE, RENDER_SIZE), _WHITE)
    draw = ImageDraw.Draw(image)
    radius = LINE_WIDTH / 2
    for stroke in strokes:
        points = [(_pixel(x), _pixel(y)) for x, y in stroke]
        if len(points) == 1:
            x, y = points[0]
            box = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(box, fill=_BLACK)
        elif points:
            draw.line(points, fill=_BLACK, width=LINE_WIDTH, joint="curve")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
