"""llama.cpp media capability probe and ``/completion`` prompt shaping (#142).

Measured on router build ``b11176-f805c57a2`` with ``gemma-3-4b-it-q4km-mm``:

- ``GET /props?model=<id>`` reports ``modalities.vision`` and a ``media_marker``
  that the server **randomizes per instance**, so the documented
  ``MEDIA_MARKER`` must be substituted before the POST.
- Only the nested object prompt ``{"prompt_string": …, "multimodal_data": […]}``
  attaches the image. A top-level ``multimodal_data`` sibling of a string
  ``prompt`` returns HTTP 200 and **silently drops** the image.
- ``multimodal_data`` entries are raw base64. A ``data:`` URI fails to load.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp.multimodal import media_prompt_field
    from typevet.domain.media import MEDIA_MARKER, ImageInput

    image = ImageInput(data=png_bytes, mime_type="image/png")
    field = media_prompt_field(
        f"{MEDIA_MARKER}Answer:",
        (image,),
        marker="<__media__>",
    )
    assert field["multimodal_data"]
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Caller of this module
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: One-retry send
      that the ``/props`` probe uses (#305)
    - [typevet.domain.media][]: ``ImageInput`` and the documented marker
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    parse_json_response,
    send_idempotent,
)
from typevet.domain.errors import GenerationError
from typevet.domain.media import MEDIA_MARKER, ImageInput


@dataclass(frozen=True, slots=True)
class MediaCapability:
    """What one router model declares about image input.

    Attributes:
        vision (bool): Whether the served model accepts image input.
        marker (str): Media placeholder this server instance expects.

    Examples:
        ```python
        from typevet.adapters.outbound.llama_cpp.multimodal import MediaCapability

        MediaCapability(vision=True, marker="<__media__>")
        ```
    """

    vision: bool
    marker: str


def _capability_from_props(payload: Any) -> MediaCapability:
    """Read ``modalities.vision`` and ``media_marker`` from a props body.

    Args:
        payload: Parsed JSON body from ``GET /props``.

    Returns:
        Declared capability; the marker falls back to ``MEDIA_MARKER`` when the
        server does not report one.

    Raises:
        GenerationError: When the props body is not a JSON object.
    """
    if not isinstance(payload, dict):
        msg = "llama.cpp props response root must be an object"
        raise GenerationError(msg)
    modalities = payload.get("modalities")
    vision = isinstance(modalities, dict) and modalities.get("vision") is True
    marker = payload.get("media_marker")
    return MediaCapability(
        vision=vision,
        marker=marker if isinstance(marker, str) and marker else MEDIA_MARKER,
    )


def fetch_media_capability(
    client: httpx.Client,
    base_url: str,
    model: str,
) -> MediaCapability:
    """Probe ``GET /props?model=<model>`` for image support and the marker.

    An early close on a new or reused connection gets one retry (#305).

    Args:
        client: Open HTTP client for the router.
        base_url: Router root with a trailing slash.
        model: Router model id to probe.

    Returns:
        The model's declared ``MediaCapability``.

    Raises:
        TransportError: When the HTTP client fails before a response.
        BackendHttpError: When the router returns HTTP status 400 or above.
        GenerationError: When the props body shape is not usable.
    """
    url = urljoin(base_url, "props")
    response = send_idempotent(lambda: client.get(url, params={"model": model}))
    ensure_success_status(response)
    return _capability_from_props(parse_json_response(response))


def media_prompt_field(
    prefix: str,
    media: tuple[ImageInput, ...],
    *,
    marker: str,
) -> dict[str, Any]:
    """Build the nested ``prompt`` object that attaches images.

    Args:
        prefix: Rendered prefix holding one ``MEDIA_MARKER`` per image.
        media: Images in prefix-marker order.
        marker: Marker this server instance expects.

    Returns:
        Object prompt with the substituted marker and raw base64 payloads.
    """
    return {
        "prompt_string": prefix.replace(MEDIA_MARKER, marker),
        "multimodal_data": [
            base64.b64encode(image.data).decode("ascii") for image in media
        ],
    }
