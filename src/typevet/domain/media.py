"""Image inputs and the media marker for image-conditioned judgment (#141).

``MEDIA_MARKER`` is the documented placeholder that callers put in a scoring
prefix, one per image. It is the llama.cpp mtmd default. Recent llama.cpp
builds randomize the marker per server instance, so an adapter may substitute
the router-reported marker at the edge; the domain contract always counts the
documented constant.

Caller text must not add a marker, or an image request would arrive with no
image (#433). A git diff, a log or a fetched page can hold the literal
``<__media__>``. ``neutralize_media_markers`` rewrites each such sequence to
``NEUTRALIZED_MEDIA_MARKER`` before the adapter adds the real markers. The
rewrite escapes the underscores, so the model still reads the original
characters. The router marker from ``GET /props`` is substituted only for the
documented constant, so neutralized text keeps no marker on either backend.
Escaping was chosen over a private marker value because the llama.cpp mtmd
default and the vLLM split both key on the documented constant.

Examples:
    ```python
    from typevet.domain.media import MEDIA_MARKER, ImageInput

    image = ImageInput(data=png_bytes, mime_type="image/png")
    prefix = f"{MEDIA_MARKER} What colour fills the image?"
    assert prefix.count(MEDIA_MARKER) == len((image,))
    ```

See Also:
    - [typevet.domain.candidate_scoring_request][]: Marker/media count invariant
    - [typevet.ports.judgment][]: Keyword-only ``media`` on ``judge``

Attributes:
    MEDIA_MARKER (str): Documented media placeholder ``<__media__>``.
    NEUTRALIZED_MEDIA_MARKER (str): The marker with a backslash before each
        underscore; it replaces a marker found in caller text.
    SUPPORTED_IMAGE_MIME_TYPES (frozenset[str]): Accepted v1 image mime types.
"""

from __future__ import annotations

from dataclasses import dataclass

from typevet.domain.errors import ScoringValidationError

MEDIA_MARKER = "<__media__>"

NEUTRALIZED_MEDIA_MARKER = r"<\_\_media\_\_>"

SUPPORTED_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
    }
)


@dataclass(frozen=True, slots=True)
class ImageInput:
    """One decoded image to condition a judgment on.

    The domain holds bytes only. URL and file resolution belongs to an adapter
    at the edge.

    Attributes:
        data (bytes): Encoded image bytes in ``mime_type`` format.
        mime_type (str): One of ``SUPPORTED_IMAGE_MIME_TYPES``, matched exactly.

    Examples:
        ```python
        from typevet.domain.media import ImageInput

        ImageInput(data=png_bytes, mime_type="image/png")
        ```
    """

    data: bytes
    mime_type: str

    def __post_init__(self) -> None:
        """Reject empty bytes and mime types outside the v1 set.

        Raises:
            ScoringValidationError: When bytes are empty or the mime type is
                not in ``SUPPORTED_IMAGE_MIME_TYPES``.
        """
        if not self.data:
            msg = "image data must be non-empty"
            raise ScoringValidationError(msg)
        if self.mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MIME_TYPES))
            msg = (
                f"unsupported image mime type {self.mime_type!r}; "
                f"supported: {supported}"
            )
            raise ScoringValidationError(msg)


def count_media_markers(text: str) -> int:
    """Count ``MEDIA_MARKER`` occurrences in a rendered prefix.

    Args:
        text: Rendered prompt prefix.

    Returns:
        Number of documented media markers in ``text``.

    Examples:
        ```python
        from typevet.domain.media import MEDIA_MARKER, count_media_markers

        assert count_media_markers(f"{MEDIA_MARKER} caption") == 1
        ```
    """
    return text.count(MEDIA_MARKER)


def neutralize_media_markers(text: str) -> str:
    """Rewrite each ``MEDIA_MARKER`` in caller text so it binds no image.

    The result holds no marker. The marker has one ``<`` and one ``>``, so
    no marker can start or end inside the escaped form.

    Args:
        text: Caller text such as a state, instructions or criteria.

    Returns:
        ``text`` with every marker replaced by ``NEUTRALIZED_MEDIA_MARKER``.

    Examples:
        ```python
        from typevet.domain.media import MEDIA_MARKER, neutralize_media_markers

        assert MEDIA_MARKER not in neutralize_media_markers(f"diff {MEDIA_MARKER}")
        ```
    """
    return text.replace(MEDIA_MARKER, NEUTRALIZED_MEDIA_MARKER)
