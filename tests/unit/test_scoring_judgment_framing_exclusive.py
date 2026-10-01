"""Unit tests: ``framing`` and ``served_template`` are mutually exclusive (#219).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_scoring_judgment_framing_exclusive.py
    ```

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: Adapter under test
"""

from __future__ import annotations

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit


class _FakeFraming:
    """Fake framing that returns a fixed prefix.

    Examples:
        ```python
        _FakeFraming().compose_prefix(user_text="u", media=())
        ```
    """

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return a plain prefix without media markers.

        Returns:
            The user text unchanged.
        """
        del media
        return user_text


def _tokenize(text: str) -> tuple[int, ...]:
    return tuple(ord(ch) for ch in text)


def test_framing_with_served_template_raises_value_error() -> None:
    """Both keywords together fail at construction and the error names both."""
    with pytest.raises(ValueError, match="framing") as excinfo:
        ScoringJudgmentAdapter(
            SequentialScoringFake([]),
            tokenize_content=_tokenize,
            framing=_FakeFraming(),
            served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
        )
    assert "served_template" in str(excinfo.value)


def test_framing_alone_and_served_template_alone_stay_valid() -> None:
    """Each keyword on its own still constructs an adapter."""
    with_framing = ScoringJudgmentAdapter(
        SequentialScoringFake([]),
        tokenize_content=_tokenize,
        framing=_FakeFraming(),
    )
    with_template = ScoringJudgmentAdapter(
        SequentialScoringFake([]),
        tokenize_content=_tokenize,
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    assert isinstance(with_framing, ScoringJudgmentAdapter)
    assert isinstance(with_template, ScoringJudgmentAdapter)
