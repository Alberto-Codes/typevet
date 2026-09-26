"""Judgment call accounting for the CORD expense live smoke ([#186][i186]).

Receipt ``requests`` and ``arm_call_counts`` must reflect every judgment
attempt, including the ``image_only`` omission control, and must sum
``model_calls`` on scored rows rather than output row counts.

Examples:
    ```python
    from typevet.evaluation.cord_expense_call_accounting import (
        cord_expense_smoke_request_totals,
    )

    total, arms = cord_expense_smoke_request_totals(
        text_only={"C1": {"model_calls": 1}},
        image_only={"R01": {"model_calls": 1}},
        combined={"C1": {"model_calls": 1}},
    )
    assert total == 4
    assert arms["image_only_omission"] == 1
    ```

See Also:
    - [tests.live.test_cord_expense_smoke_live][]: live wiring
    - [typevet.evaluation.cord_expense_receipt_requirement][]: deterministic rows

[i186]: https://github.com/Alberto-Codes/typevet/issues/186
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_OMISSION_ARM = "image_only_omission"


def _judgment_calls_in_arm(rows: Mapping[str, Mapping[str, Any]]) -> int:
    total = 0
    for row in rows.values():
        calls = row.get("model_calls", 1)
        if not isinstance(calls, int):
            msg = "each scored row must report integer model_calls"
            raise TypeError(msg)
        total += calls
    return total


def summarize_cord_expense_judgment_calls(
    *,
    text_only: Mapping[str, Mapping[str, Any]],
    image_only: Mapping[str, Mapping[str, Any]],
    combined: Mapping[str, Mapping[str, Any]],
    image_only_omission_calls: int,
) -> tuple[int, dict[str, int]]:
    """Return total judgment attempts and per-arm counts for one smoke run.

    Args:
        text_only: ``text_only`` rows keyed by claim id.
        image_only: ``image_only`` rows keyed by receipt id.
        combined: ``combined`` rows keyed by claim id.
        image_only_omission_calls: Judgment attempts for the omission control
            (same ``image_only`` prompt without media).

    Returns:
        Total judgment calls and arm_call_counts for experiment identity.

    Raises:
        TypeError: When a row omits integer ``model_calls``.
        ValueError: When ``image_only_omission_calls`` is invalid.
    """
    if not isinstance(image_only_omission_calls, int) or image_only_omission_calls < 0:
        msg = "image_only_omission_calls must be a non-negative int"
        raise ValueError(msg)
    arms = {
        "text_only": _judgment_calls_in_arm(text_only),
        "image_only": _judgment_calls_in_arm(image_only),
        "combined": _judgment_calls_in_arm(combined),
        _OMISSION_ARM: image_only_omission_calls,
    }
    return sum(arms.values()), arms


def cord_expense_smoke_request_totals(
    *,
    text_only: Mapping[str, Mapping[str, Any]],
    image_only: Mapping[str, Mapping[str, Any]],
    combined: Mapping[str, Mapping[str, Any]],
) -> tuple[int, dict[str, int]]:
    """Return receipt ``requests`` and arm counts for one CORD expense smoke run.

    The live smoke always runs one ``image_only`` omission baseline judgment after
    the scored arms. Callers must use this helper instead of
    ``summarize_cord_expense_judgment_calls`` so omission accounting stays wired.

    Args:
        text_only: ``text_only`` rows keyed by claim id.
        image_only: ``image_only`` rows keyed by receipt id.
        combined: ``combined`` rows keyed by claim id.

    Returns:
        Total judgment calls and per-arm counts for experiment identity.
    """
    return summarize_cord_expense_judgment_calls(
        text_only=text_only,
        image_only=image_only,
        combined=combined,
        image_only_omission_calls=1,
    )
