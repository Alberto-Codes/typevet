"""Gemma served-template and answer-binding adapters (#148).

Examples:
    ```python
    from typevet.adapters.outbound.gemma import (
        CHATML_ASSISTANT_HEADER,
        classify_served_template,
        resolve_answer_anchor,
    )
    ```

See Also:
    - [typevet.gemma_served_template][]: Root compatibility shim
    - [typevet.gemma_answer_binding][]: Root compatibility shim

Attributes:
    CHATML_ASSISTANT_HEADER (str): ChatML assistant turn header constant.
    ServedTemplateClass (type): Served-template family enum.
    ThinkingDisposition (type): Thinking-channel disposition enum.
    classify_served_template (callable): Classify ``/apply-template`` output.
    compose_media_scoring_prefix (callable): Native-turn prefix for media scoring.
    resolve_answer_anchor (callable): Locate pre-candidate answer boundary.
"""

from typevet.adapters.outbound.gemma.answer_binding import (
    AnswerAnchor,
    ThinkingDisposition,
    bind_enum_label,
    bind_enum_labels,
    resolve_answer_anchor,
    termination_kind,
)
from typevet.adapters.outbound.gemma.scoring_prefix import (
    compose_media_scoring_prefix,
    compose_scoring_prefix,
)
from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
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
    "GEMMA3_END_OF_TURN",
    "GEMMA3_MODEL_TURN_HEADER",
    "GEMMA3_START_OF_TURN",
    "GEMMA4_CHANNEL_CLOSE",
    "GEMMA4_MODEL_TURN_HEADER",
    "GEMMA4_NO_THINKING_PREFILL",
    "GEMMA4_THINK_TRIGGER",
    "GEMMA4_TOOL_RESPONSE",
    "GEMMA4_TURN_CLOSE",
    "GEMMA4_TURN_OPEN",
    "AnswerAnchor",
    "ServedTemplateClass",
    "ThinkingDisposition",
    "bind_enum_label",
    "bind_enum_labels",
    "classify_served_template",
    "compose_media_scoring_prefix",
    "compose_scoring_prefix",
    "label_embeds_control_fragment",
    "resolve_answer_anchor",
    "stop_markers_for",
    "termination_kind",
]
