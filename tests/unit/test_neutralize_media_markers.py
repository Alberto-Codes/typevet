"""Unit tests for ``neutralize_media_markers`` (#433).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_neutralize_media_markers.py
    ```

See Also:
    - [typevet.domain.media][]: The function under test
"""

from __future__ import annotations

import pytest

from typevet.domain.media import (
    MEDIA_MARKER,
    count_media_markers,
    neutralize_media_markers,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        MEDIA_MARKER,
        MEDIA_MARKER * 3,
        f"<{MEDIA_MARKER}>",
        "<<__media__>__media__>",
        "<__media<__media__>__>",
        f"a{MEDIA_MARKER}b\n{MEDIA_MARKER}",
    ],
)
def test_neutralized_text_holds_no_marker(text: str) -> None:
    assert count_media_markers(neutralize_media_markers(text)) == 0


def test_text_without_marker_is_unchanged() -> None:
    text = "plain <__media> and __media__ text"
    assert neutralize_media_markers(text) == text


def test_neutralized_marker_stays_readable() -> None:
    assert neutralize_media_markers(MEDIA_MARKER) == r"<\_\_media\_\_>"
