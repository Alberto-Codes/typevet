"""Compatibility shim for field instructions and scoring-prefix compose.

Prefer importing from ``typevet.domain.field_instructions`` and
``typevet.adapters.outbound.gemma.scoring_prefix`` (or ``typevet.runtime``)
in new code.

Examples:
    ```python
    from typevet.field_prompt import compose_scoring_prefix, render_field_instructions
    ```

See Also:
    - [typevet.domain.field_instructions][]: Model-agnostic field blocks
    - [typevet.adapters.outbound.gemma.scoring_prefix][]: ChatML prefix compose
    - [typevet.runtime.scoring_prefix][]: Runtime re-export of compose
"""

from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix
from typevet.domain.field_instructions import (
    choice_criteria_from_schema,
    gold_reference_markers,
    render_field_instructions,
)

__all__ = [
    "choice_criteria_from_schema",
    "compose_scoring_prefix",
    "gold_reference_markers",
    "render_field_instructions",
]
