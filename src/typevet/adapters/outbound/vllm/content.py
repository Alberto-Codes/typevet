"""Chat content blocks for vLLM requests that carry images.

vLLM takes one user message whose content is a list of ``text`` and
``image_url`` blocks. Each ``MEDIA_MARKER`` in the text becomes one
``image_url`` block with a ``data:`` URI, in ``media`` order. The scoring and
generation adapters share this builder.

Examples:
    ```python
    from typevet.adapters.outbound.vllm.content import content_blocks
    from typevet.domain.media import MEDIA_MARKER, ImageInput

    image = ImageInput(data=b"png", mime_type="image/png")
    blocks = content_blocks(f"Describe: {MEDIA_MARKER}", (image,))
    assert blocks[0] == {"type": "text", "text": "Describe: "}
    assert blocks[1]["type"] == "image_url"
    ```

See Also:
    - [typevet.adapters.outbound.vllm.scoring][]: Candidate scoring adapter
    - [typevet.adapters.outbound.vllm.generation][]: Typed generation adapter
    - [typevet.domain.media][]: MEDIA_MARKER and ImageInput
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from typevet.domain.media import MEDIA_MARKER

if TYPE_CHECKING:
    from typevet.domain.media import ImageInput

__all__ = ["content_blocks"]


def content_blocks(prefix: str, media: tuple[ImageInput, ...]) -> list[dict[str, Any]]:
    """Split a prefix at its media markers into chat content blocks.

    Each marker becomes one ``image_url`` block, in ``media`` order. The text
    between markers becomes ``text`` blocks, verbatim except that one newline
    directly after a marker is dropped. Empty text blocks are left out.

    Args:
        prefix: Text that holds one ``MEDIA_MARKER`` per image.
        media: Images the markers stand for, in marker order.

    Returns:
        Content blocks in prefix order.
    """
    parts = prefix.split(MEDIA_MARKER)
    blocks: list[dict[str, Any]] = []
    if parts[0]:
        blocks.append({"type": "text", "text": parts[0]})
    for image, part in zip(media, parts[1:], strict=True):
        encoded = base64.b64encode(image.data).decode("ascii")
        url = f"data:{image.mime_type};base64,{encoded}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
        text = part.removeprefix("\n")
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks
