"""Curated Quick, Draw! categories for the doodle ``Choice`` (#412).

The 24 names are present in Google's official ``categories.txt``. The list
order is the ``Choice`` option order. Ordinal controls bind at most 36
labels, so the list stays at or below that limit.

Attributes:
    DOODLE_CATEGORIES (tuple[str, ...]): Category names in option order.

Examples:
    ```python
    from typevet_evals.doodle_duel.categories import DOODLE_CATEGORIES

    assert "cat" in DOODLE_CATEGORIES
    ```

See Also:
    - [typevet_evals.datasets.quickdraw][]: the doodles of each category
"""

from __future__ import annotations

from typing import Final

DOODLE_CATEGORIES: Final[tuple[str, ...]] = (
    "cat",
    "pizza",
    "castle",
    "sword",
    "bicycle",
    "apple",
    "house",
    "tree",
    "fish",
    "car",
    "sun",
    "star",
    "umbrella",
    "guitar",
    "airplane",
    "banana",
    "clock",
    "eyeglasses",
    "flower",
    "key",
    "ladder",
    "moon",
    "snowman",
    "spider",
)
