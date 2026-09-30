"""Deterministic PNG renders of synthetic checks and a contact sheet (#315).

:func:`render_check` draws one :class:`~typevet_evals.check_match.cases.CheckCase`
with Pillow's bundled default font only (amendment A3). Each render carries
a ``SPECIMEN`` mark across the face and a ``VOID`` mark across the signature
line. Signatures are seeded synthetic strokes. The repository stores no
render: callers keep renders outside the repository.

Renders are byte-identical for the same case within one Pillow and FreeType
build. A different build can change the bytes.

Attributes:
    CHECK_SIZE (tuple[int, int]): Width and height of one render, in pixels.
    BANK_NAME (str): Invented bank name printed on every check.
    DRAWER_LINES (tuple[str, str]): Invented drawer name and address.
    SPECIMEN_MARK (str): Mark across the face.
    VOID_MARK (str): Mark across the signature line.

Examples:
    ```python
    from typevet_evals.check_match import check_cases, render_check

    png = render_check(check_cases()[0])
    assert png[1:4] == b"PNG"
    ```

See Also:
    - [typevet_evals.check_match.cases][]: the cases that this module draws
    - [typevet_evals.check_match.words][]: the written amount
"""

from __future__ import annotations

import io
import math
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from typevet_evals.check_match.cases import CheckCase, CheckFace, SeededDraws
from typevet_evals.check_match.words import amount_in_words

CHECK_SIZE: Final[tuple[int, int]] = (1000, 440)
BANK_NAME: Final[str] = "SYNTHETIC TEST BANK - NOT A REAL BANK"
DRAWER_LINES: Final[tuple[str, str]] = ("Sample Holder", "100 Example Road, Testville")
SPECIMEN_MARK: Final[str] = "SPECIMEN"
VOID_MARK: Final[str] = "VOID"

_MONTHS: Final[tuple[str, ...]] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_PAPER: Final[tuple[int, int, int]] = (246, 243, 232)
_INK: Final[tuple[int, int, int]] = (25, 25, 30)
_PEN: Final[tuple[int, int, int]] = (20, 40, 120)
_SIGNATURE_BOX: Final[tuple[int, int, int, int]] = (600, 300, 940, 350)
_CAPTION_HEIGHT: Final[int] = 64
_CAPTION_CHARS: Final[int] = 78


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return Pillow's bundled default font at ``size`` pixels.

    Args:
        size: Font size in pixels.

    Returns:
        The default font.
    """
    return ImageFont.load_default(size=size)


def _printed_date(face: CheckFace) -> str:
    """Return the check date as ``Month D, YYYY`` without the locale.

    Args:
        face: Check face.

    Returns:
        The date text.
    """
    day = face.date
    return f"{_MONTHS[day.month - 1]} {day.day}, {day.year}"


def _dollars(cents: int) -> str:
    """Return ``cents`` as ``1,234.56``.

    Args:
        cents: Amount in cents.

    Returns:
        The amount text.
    """
    whole, rest = divmod(cents, 100)
    return f"{whole:,}.{rest:02d}"


def _draw_fields(draw: ImageDraw.ImageDraw, face: CheckFace) -> None:
    """Draw the printed fields, lines and labels of one check.

    Args:
        draw: Drawing context of the check image.
        face: What the check shows.
    """
    width, height = CHECK_SIZE
    small, label, body = _font(14), _font(16), _font(22)
    draw.rectangle((6, 6, width - 7, height - 7), outline=_INK, width=2)
    draw.text((30, 24), DRAWER_LINES[0], font=body, fill=_INK)
    draw.text((30, 52), DRAWER_LINES[1], font=small, fill=_INK)
    draw.text((330, 24), BANK_NAME, font=label, fill=_INK)
    draw.text((880, 24), str(face.check_number), font=body, fill=_INK)
    draw.text((640, 90), "DATE", font=label, fill=_INK)
    draw.text((700, 84), _printed_date(face), font=body, fill=_INK)
    draw.line((695, 112, 940, 112), fill=_INK, width=1)
    draw.text((30, 150), "PAY TO THE", font=small, fill=_INK)
    draw.text((30, 166), "ORDER OF", font=small, fill=_INK)
    draw.text((130, 152), face.payee, font=body, fill=_INK)
    draw.line((125, 182, 740, 182), fill=_INK, width=1)
    draw.rectangle((770, 146, 940, 186), outline=_INK, width=2)
    draw.text((780, 154), f"$ {_dollars(face.numeric_cents)}", font=body, fill=_INK)
    words = f"{amount_in_words(face.written_cents)} DOLLARS"
    draw.text((40, 212), words, font=body, fill=_INK)
    draw.line((30, 242, 940, 242), fill=_INK, width=1)
    draw.text((30, 330), "MEMO", font=label, fill=_INK)
    draw.line((85, 348, 420, 348), fill=_INK, width=1)
    left, _, right, bottom = _SIGNATURE_BOX
    draw.line((left, bottom, right, bottom), fill=_INK, width=1)
    draw.text((left, bottom + 4), "AUTHORIZED SIGNATURE", font=small, fill=_INK)
    micr = (
        f"ROUTING {face.routing_number}   ACCOUNT {face.account_number}   "
        f"CHECK {face.check_number}"
    )
    draw.text((30, 392), micr, font=label, fill=_INK)
    draw.text((640, 392), "NON-NEGOTIABLE TEST IMAGE", font=small, fill=_INK)


def _draw_signature(draw: ImageDraw.ImageDraw, seed: int) -> None:
    """Draw seeded synthetic pen strokes above the signature line.

    Args:
        draw: Drawing context of the check image.
        seed: Seed of the strokes; never a real signature.
    """
    draws = SeededDraws(f"check-match:signature:{seed}")
    left, top, right, bottom = _SIGNATURE_BOX
    x = left + 20 + draws.integer(0, 30)
    for _ in range(draws.integer(2, 4)):
        span = draws.integer(60, 110)
        amplitude = draws.uniform(6.0, 16.0)
        frequency = draws.uniform(0.08, 0.2)
        phase = draws.uniform(0.0, math.tau)
        base = (top + bottom) / 2 + draws.uniform(-6.0, 6.0)
        points = [
            (x + step, base + amplitude * math.sin(frequency * step + phase))
            for step in range(0, span, 3)
        ]
        draw.line(points, fill=_PEN, width=3, joint="curve")
        x = min(x + span + draws.integer(5, 20), right - 110)


def _mark_layer(text: str, size: int, fill: tuple[int, int, int, int]) -> Image.Image:
    """Return an RGBA image holding one mark text.

    Args:
        text: Mark text.
        size: Font size in pixels.
        fill: RGBA colour of the text.

    Returns:
        A transparent image just large enough for the text.
    """
    font = _font(size)
    left, top, right, bottom = font.getbbox(text)
    layer = Image.new("RGBA", (int(right - left) + 8, int(bottom - top) + 8))
    ImageDraw.Draw(layer).text((4 - left, 4 - top), text, font=font, fill=fill)
    return layer


def _draw_marks(image: Image.Image) -> Image.Image:
    """Overlay ``SPECIMEN`` across the face and ``VOID`` across the signature.

    Args:
        image: RGB check image.

    Returns:
        A new RGB image with both marks.
    """
    canvas = image.convert("RGBA")
    specimen = _mark_layer(SPECIMEN_MARK, 120, (120, 120, 120, 70)).rotate(
        12, expand=True, resample=Image.Resampling.BICUBIC
    )
    width, height = CHECK_SIZE
    overlay = Image.new("RGBA", canvas.size)
    overlay.paste(
        specimen, ((width - specimen.width) // 2, (height - specimen.height) // 2)
    )
    void = _mark_layer(VOID_MARK, 60, (190, 30, 30, 90))
    left, _, right, bottom = _SIGNATURE_BOX
    overlay.paste(
        void, ((left + right - void.width) // 2, bottom - void.height // 2 - 8)
    )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def render_check_image(case: CheckCase) -> Image.Image:
    """Return the rendered check for ``case`` as a Pillow image.

    Args:
        case: Check case.

    Returns:
        An RGB image of :data:`CHECK_SIZE`, blurred when the face sets a
        blur radius.
    """
    image = Image.new("RGB", CHECK_SIZE, _PAPER)
    draw = ImageDraw.Draw(image)
    _draw_fields(draw, case.face)
    if case.face.signed:
        _draw_signature(draw, case.face.signature_seed)
    image = _draw_marks(image)
    if case.face.blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(case.face.blur_radius))
    return image


def _png_bytes(image: Image.Image) -> bytes:
    """Encode ``image`` as PNG with pinned options and no metadata.

    Args:
        image: Image to encode.

    Returns:
        PNG bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def render_check(case: CheckCase) -> bytes:
    """Return the rendered check for ``case`` as PNG bytes.

    Args:
        case: Check case.

    Returns:
        PNG bytes; the same case gives the same bytes within one Pillow
        build.
    """
    return _png_bytes(render_check_image(case))


def _caption(case: CheckCase) -> str:
    """Return the contact-sheet caption for ``case``.

    Args:
        case: Check case.

    Returns:
        Case id, expected labels, blur radius and the register row text.
    """
    expected = case.expected
    verdicts = "/".join(sorted(expected.accepted_verdicts))
    head = (
        f"{case.case_id}  expect {verdicts}  payee={expected.payee_matches} "
        f"amounts={expected.amounts_match}"
    )
    if case.face.blur_radius:
        head += f"  blur={case.face.blur_radius}"
    return "\n".join(
        [head, *textwrap.wrap("register: " + case.row.as_text(), _CAPTION_CHARS)]
    )


def write_contact_sheet(
    cases: Sequence[CheckCase], path: Path, *, columns: int = 7, scale: float = 0.5
) -> Path:
    """Write one PNG grid of renders with their expected labels.

    Args:
        cases: Cases to draw, in grid order.
        path: Output file; the caller chooses a path outside the repository.
        columns: Renders per grid row.
        scale: Render scale in the grid.

    Returns:
        ``path``, after the file is written.

    Raises:
        ValueError: When ``cases`` is empty.
    """
    if not cases:
        msg = "contact sheet needs at least one case"
        raise ValueError(msg)
    cell_w, cell_h = int(CHECK_SIZE[0] * scale), int(CHECK_SIZE[1] * scale)
    rows = math.ceil(len(cases) / columns)
    width = min(columns, len(cases)) * cell_w
    sheet = Image.new("RGB", (width, rows * (cell_h + _CAPTION_HEIGHT)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(12)
    for position, case in enumerate(cases):
        row, column = divmod(position, columns)
        x, y = column * cell_w, row * (cell_h + _CAPTION_HEIGHT)
        thumb = render_check_image(case).resize(
            (cell_w, cell_h), Image.Resampling.LANCZOS
        )
        sheet.paste(thumb, (x, y))
        draw.multiline_text(
            (x + 4, y + cell_h + 4), _caption(case), font=font, fill="black"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(sheet))
    return path
