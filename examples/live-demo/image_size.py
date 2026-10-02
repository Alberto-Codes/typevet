"""Image size sniffers for the live demo's behind-the-scenes panel.

The panel names the image as ``name, width x height, sha256``. These readers
take the size from the PNG, JPEG or WebP header bytes, without Pillow, so the
size is the size of the bytes that typevet sends.

``judge.py`` imports this module from the same directory.

Examples:
    ```python
    size = image_size(data, "image/png")
    ```

See Also:
    - [typevet.domain.ImageInput][]: The typed image input.
    - examples/live-demo/README.md: How to run the live demo.
"""

from __future__ import annotations

import struct

JPEG_MARKER = 0xFF
JPEG_SOF_FIRST = 0xC0
JPEG_SOF_LAST = 0xCF
JPEG_NOT_SOF = (0xC4, 0xC8, 0xCC)


def _png_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from a PNG ``IHDR`` chunk.

    Args:
        data: Image bytes.

    Returns:
        The size, or None when the header is not ``IHDR``.
    """
    if data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from the first JPEG start-of-frame segment.

    Args:
        data: Image bytes.

    Returns:
        The size, or None when no start-of-frame segment is found.
    """
    i = 2
    while i + 9 < len(data):
        if data[i] != JPEG_MARKER:
            i += 1
            continue
        m = data[i + 1]
        seg = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if JPEG_SOF_FIRST <= m <= JPEG_SOF_LAST and m not in JPEG_NOT_SOF:
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        i += 2 + seg
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) from a WebP ``VP8X``, ``VP8`` or ``VP8L`` chunk.

    Args:
        data: Image bytes.

    Returns:
        The size, or None for another chunk kind.
    """
    kind = data[12:16]
    if kind == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if kind == b"VP8 ":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        b = int.from_bytes(data[21:25], "little")
        return (b & 0x3FFF) + 1, ((b >> 14) & 0x3FFF) + 1
    return None


def image_size(data: bytes, mime: str) -> tuple[int, int] | None:
    """Return (width, height) for PNG, JPEG or WebP bytes, else None.

    Args:
        data: Image bytes.
        mime: Image mime type.

    Returns:
        The size, or None when the type or the header is not known.
    """
    readers = {
        "image/png": _png_size,
        "image/jpeg": _jpeg_size,
        "image/webp": _webp_size,
    }
    reader = readers.get(mime)
    try:
        return reader(data) if reader is not None else None
    except (struct.error, IndexError):
        return None
