"""Synthetic single-colour PNG fixtures for image-conditioned judgment.

Each image is **synthetic**: the author picks the fill colour, so the gold
label of a "what colour fills this image" question is known without any
third-party corpus. Built with ``zlib`` and ``struct`` so the test suite needs
no image library.
"""

from __future__ import annotations

import struct
import zlib

from typevet.domain.media import ImageInput

FILL_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def solid_png(rgb: tuple[int, int, int], *, size: int = 32) -> bytes:
    """Encode a ``size``x``size`` truecolour PNG filled with one colour.

    Args:
        rgb: Red, green and blue channel values from 0 to 255.
        size: Square edge length in pixels.

    Returns:
        Complete PNG bytes.
    """
    scanline = b"\x00" + bytes(rgb) * size
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(scanline * size, 9))
        + _chunk(b"IEND", b"")
    )


def solid_image(colour: str) -> ImageInput:
    """Build an ``ImageInput`` filled with one named colour.

    Args:
        colour: Key of ``FILL_COLOURS``.

    Returns:
        ``ImageInput`` holding a synthetic PNG.

    Raises:
        KeyError: When ``colour`` is not a known fill colour.
    """
    return ImageInput(data=solid_png(FILL_COLOURS[colour]), mime_type="image/png")
