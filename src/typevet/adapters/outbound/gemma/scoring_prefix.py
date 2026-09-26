"""Degraded ChatML scoring-prefix composition for Gemma judgment (#148).

Examples:
    ```python
    from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix
    ```

See Also:
    - [typevet.domain.field_instructions][]: Model-agnostic field blocks
    - [typevet.runtime.scoring_prefix][]: Runtime re-export
"""

from __future__ import annotations

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
)


def compose_scoring_prefix(*, context: str, field_block: str) -> str:
    """Compose degraded ChatML text ending at the assistant answer boundary.

    Args:
        context: Caller-owned user or task text for the judgment.
        field_block: Rendered field instructions from ``render_field_instructions``.

    Returns:
        Prefix string ending with ``CHATML_ASSISTANT_HEADER``.
    """
    return (
        f"{CHATML_IM_START}user\n{context}\n\n{field_block}{CHATML_IM_END}\n"
        f"{CHATML_ASSISTANT_HEADER}"
    )
