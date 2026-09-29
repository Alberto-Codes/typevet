"""Unit tests for image media inputs and the scoring marker invariant (#141/#142)."""

from __future__ import annotations

from typing import Any

import pytest

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import ScoringValidationError
from typevet.domain.media import (
    MEDIA_MARKER,
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageInput,
    count_media_markers,
)
from typevet.domain.models import GenerationRequest

_PNG = b"\x89PNG\r\n\x1a\nfake"


def _request(
    *,
    prefix: str,
    media: tuple[ImageInput, ...] = (),
) -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model="gemma-test",
        prefix=prefix,
        candidates=(
            CandidateTokenSpec("a", (101,)),
            CandidateTokenSpec("b", (202,)),
        ),
        media=media,
    )


@pytest.mark.unit
def test_media_marker_constant_is_the_llama_cpp_default() -> None:
    assert MEDIA_MARKER == "<__media__>"


@pytest.mark.unit
@pytest.mark.parametrize("mime_type", sorted(SUPPORTED_IMAGE_MIME_TYPES))
def test_image_input_accepts_supported_mime_types(mime_type: str) -> None:
    image = ImageInput(data=_PNG, mime_type=mime_type)
    assert image.mime_type == mime_type
    assert image.data == _PNG


@pytest.mark.unit
def test_supported_mime_types_are_exactly_v1_set() -> None:
    assert sorted(SUPPORTED_IMAGE_MIME_TYPES) == [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]


@pytest.mark.unit
def test_image_input_rejects_empty_bytes() -> None:
    with pytest.raises(ScoringValidationError, match="non-empty"):
        ImageInput(data=b"", mime_type="image/png")


@pytest.mark.unit
@pytest.mark.parametrize(
    "mime_type",
    ["image/gif", "image/svg+xml", "text/plain", "", "  ", "IMAGE/PNG "],
    ids=["gif", "svg", "text", "empty", "blank", "padded-uppercase"],
)
def test_image_input_rejects_unknown_mime_type(mime_type: str) -> None:
    with pytest.raises(ScoringValidationError, match="mime type"):
        ImageInput(data=_PNG, mime_type=mime_type)


@pytest.mark.unit
def test_count_media_markers_counts_every_occurrence() -> None:
    assert count_media_markers("no media here") == 0
    assert count_media_markers(f"{MEDIA_MARKER} text") == 1
    assert count_media_markers(f"{MEDIA_MARKER}{MEDIA_MARKER}") == 2


@pytest.mark.unit
def test_scoring_request_defaults_to_no_media() -> None:
    request = _request(prefix="Answer:")
    assert request.media == ()


@pytest.mark.unit
def test_scoring_request_rejects_media_without_marker() -> None:
    image = ImageInput(data=_PNG, mime_type="image/png")
    with pytest.raises(ScoringValidationError, match="media marker"):
        _request(prefix="Answer:", media=(image,))


@pytest.mark.unit
def test_scoring_request_rejects_marker_count_mismatch() -> None:
    image = ImageInput(data=_PNG, mime_type="image/png")
    with pytest.raises(ScoringValidationError, match="media marker"):
        _request(prefix=f"{MEDIA_MARKER}{MEDIA_MARKER}\nAnswer:", media=(image,))


@pytest.mark.unit
def test_scoring_request_rejects_marker_without_media() -> None:
    with pytest.raises(ScoringValidationError, match="media marker"):
        _request(prefix=f"{MEDIA_MARKER}\nAnswer:")


@pytest.mark.unit
def test_scoring_request_accepts_matched_marker_count() -> None:
    images = (
        ImageInput(data=_PNG, mime_type="image/png"),
        ImageInput(data=_PNG, mime_type="image/jpeg"),
    )
    request = _request(
        prefix=f"{MEDIA_MARKER}\n{MEDIA_MARKER}\nAnswer:",
        media=images,
    )
    assert request.media == images


_OBJECT_SCHEMA = {"type": "object"}


@pytest.mark.unit
def test_generation_request_stores_list_media_as_tuple() -> None:
    image = ImageInput(data=_PNG, mime_type="image/png")
    media: Any = [image]
    request = GenerationRequest(
        prompt=f"{MEDIA_MARKER}\nDescribe.",
        schema=_OBJECT_SCHEMA,
        model="gemma-test",
        media=media,
    )
    assert isinstance(request.media, tuple)
    assert request.media == (image,)


@pytest.mark.unit
def test_generation_request_rejects_non_image_media_item() -> None:
    media: Any = [b"x"]
    with pytest.raises(TypeError, match=r"media\[0\].*bytes"):
        GenerationRequest(
            prompt=f"{MEDIA_MARKER}\nDescribe.",
            schema=_OBJECT_SCHEMA,
            model="gemma-test",
            media=media,
        )
