"""Compatibility shim for Gemma served-template classification (#148).

Examples:
    ```python
    from typevet.gemma_served_template import classify_served_template
    ```

See Also:
    - [typevet.adapters.outbound.gemma.served_template][]: Canonical module
"""

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA4_CHANNEL_CLOSE,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_THINK_TRIGGER,
    GEMMA4_TOOL_RESPONSE,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
    classify_served_template,
    label_embeds_control_fragment,
    stop_markers_for,
)

__all__ = [
    "CHATML_ASSISTANT_HEADER",
    "CHATML_IM_END",
    "CHATML_IM_START",
    "GEMMA4_CHANNEL_CLOSE",
    "GEMMA4_MODEL_TURN_HEADER",
    "GEMMA4_NO_THINKING_PREFILL",
    "GEMMA4_THINK_TRIGGER",
    "GEMMA4_TOOL_RESPONSE",
    "GEMMA4_TURN_CLOSE",
    "GEMMA4_TURN_OPEN",
    "ServedTemplateClass",
    "classify_served_template",
    "label_embeds_control_fragment",
    "stop_markers_for",
]
