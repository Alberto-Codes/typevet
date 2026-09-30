"""Unit tests for CandidateScoringRequest media coercion and item checks."""

from __future__ import annotations

from typing import Any

import pytest

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.media import MEDIA_MARKER, ImageInput

_PNG = b"\x89PNG\r\n\x1a\nfake"
_CANDIDATES = (CandidateTokenSpec("a", (1,)), CandidateTokenSpec("b", (2,)))


@pytest.mark.unit
def test_scoring_request_stores_list_media_as_tuple() -> None:
    image = ImageInput(data=_PNG, mime_type="image/png")
    media: Any = [image]
    request = CandidateScoringRequest(
        model="m",
        prefix=f"{MEDIA_MARKER}\nPick:",
        candidates=_CANDIDATES,
        media=media,
    )
    assert isinstance(request.media, tuple)
    assert request.media == (image,)


@pytest.mark.unit
def test_scoring_request_rejects_non_image_media_item() -> None:
    media: Any = [b"x"]
    with pytest.raises(TypeError, match=r"media\[0\] must be ImageInput, got bytes"):
        CandidateScoringRequest(
            model="m",
            prefix=f"{MEDIA_MARKER}\nPick:",
            candidates=_CANDIDATES,
            media=media,
        )
