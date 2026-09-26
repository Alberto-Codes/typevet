"""Compatibility shim for Gemma answer binding (#148).

Examples:
    ```python
    from typevet.gemma_answer_binding import ThinkingDisposition
    ```

See Also:
    - [typevet.adapters.outbound.gemma.answer_binding][]: Canonical module
"""

from typevet.adapters.outbound.gemma.answer_binding import (
    AnswerAnchor,
    ThinkingDisposition,
    bind_enum_label,
    bind_enum_labels,
    resolve_answer_anchor,
    termination_kind,
)

__all__ = [
    "AnswerAnchor",
    "ThinkingDisposition",
    "bind_enum_label",
    "bind_enum_labels",
    "resolve_answer_anchor",
    "termination_kind",
]
