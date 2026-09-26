"""Reject non-finite floats in parsed structured generation values.

Examples:
    ```python
    from typevet.adapters.outbound.generation_finite import (
        reject_non_finite_numbers,
    )

    reject_non_finite_numbers({"score": 1.0})
    ```

See Also:
    - [typevet.domain.errors][]: SchemaValidationError
"""

from __future__ import annotations

import math
from typing import Any

from typevet.domain.errors import SchemaValidationError


def reject_non_finite_numbers(value: Any) -> None:
    """Raise when a non-finite float appears anywhere in a JSON-like tree.

    Args:
        value: Parsed model output (objects, arrays, numbers, strings, etc.).

    Raises:
        SchemaValidationError: When a float is NaN or positive/negative infinity.
    """
    _reject_non_finite(value, root=value)


def _reject_non_finite(node: Any, *, root: Any) -> None:
    if isinstance(node, float):
        if not math.isfinite(node):
            raise SchemaValidationError(
                "structured output contains a non-finite number",
                payload=root,
            )
        return
    if isinstance(node, dict):
        for item in node.values():
            _reject_non_finite(item, root=root)
        return
    if isinstance(node, list):
        for item in node:
            _reject_non_finite(item, root=root)
