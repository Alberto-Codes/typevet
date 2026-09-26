r"""Classify llama.cpp ``/apply-template`` families for Gemma judgment scoring.

Recognized families are native Gemma 4 ``<|turn>``, native Gemma 3
``<start_of_turn>`` and degraded ChatML. Mixed markers are unsupported.

Examples:
    ```python
    from typevet.adapters.outbound.gemma.served_template import classify_served_template

    assert (
        classify_served_template("<|im_start|>assistant\n").value == "degraded_chatml"
    )
    ```

See Also:
    - [typevet.adapters.outbound.gemma.answer_binding][]: Answer-prefix anchors
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

GEMMA4_TURN_OPEN: Final[str] = "<|turn>"
GEMMA4_TURN_CLOSE: Final[str] = "<turn|>"
GEMMA4_MODEL_TURN_HEADER: Final[str] = f"{GEMMA4_TURN_OPEN}model\n"
GEMMA4_NO_THINKING_PREFILL: Final[str] = "<|channel>thought\n<channel|>"
CHATML_IM_START: Final[str] = "<|im_start|>"
CHATML_IM_END: Final[str] = "<|" + "im_end|>"
CHATML_ASSISTANT_HEADER: Final[str] = f"{CHATML_IM_START}assistant\n"
GEMMA4_CHANNEL_CLOSE: Final[str] = "<channel|>"
GEMMA4_TOOL_RESPONSE: Final[str] = "<|tool_response>"
GEMMA4_THINK_TRIGGER: Final[str] = "<|think|>"
GEMMA3_START_OF_TURN: Final[str] = "<start_of_turn>"
GEMMA3_END_OF_TURN: Final[str] = "<end_of_turn>"
GEMMA3_MODEL_TURN_HEADER: Final[str] = f"{GEMMA3_START_OF_TURN}model\n"
_CONTROL_LABEL_SPLITTERS: Final[frozenset[str]] = frozenset(
    {
        CHATML_IM_START,
        CHATML_IM_END,
        GEMMA3_START_OF_TURN,
        GEMMA3_END_OF_TURN,
        GEMMA4_TURN_OPEN,
        GEMMA4_TURN_CLOSE,
        GEMMA4_CHANNEL_CLOSE,
        GEMMA4_TOOL_RESPONSE,
        GEMMA4_THINK_TRIGGER,
        "<|channel>",
    }
)


class ServedTemplateClass(StrEnum):
    """Served-template families recognized for Gemma judgment scoring.

    Attributes:
        NATIVE_GEMMA4_TURN (ServedTemplateClass): Official `<|turn>` family.
        NATIVE_GEMMA3_TURN (ServedTemplateClass): Gemma 3 `<start_of_turn>` family.
        DEGRADED_CHATML (ServedTemplateClass): ChatML-like served template.
        UNSUPPORTED (ServedTemplateClass): Mixed or unrecognized markers.

    Examples:
        ```python
        assert ServedTemplateClass.DEGRADED_CHATML.value == "degraded_chatml"
        ```
    """

    NATIVE_GEMMA4_TURN = "native_gemma4_turn"
    NATIVE_GEMMA3_TURN = "native_gemma3_turn"
    DEGRADED_CHATML = "degraded_chatml"
    UNSUPPORTED = "unsupported"


def classify_served_template(rendered_prompt: str) -> ServedTemplateClass:
    """Classify ``/apply-template`` output into a served-template family.

    Args:
        rendered_prompt: Full rendered chat prompt from the server.

    Returns:
        One of native Gemma 4 turn, native Gemma 3 turn, degraded ChatML, or
        unsupported when markers from more than one family appear.
    """
    has_turn = (
        GEMMA4_TURN_OPEN in rendered_prompt or GEMMA4_TURN_CLOSE in rendered_prompt
    )
    has_gemma3 = (
        GEMMA3_START_OF_TURN in rendered_prompt or GEMMA3_END_OF_TURN in rendered_prompt
    )
    has_chatml = CHATML_IM_START in rendered_prompt
    families = {
        ServedTemplateClass.NATIVE_GEMMA4_TURN: has_turn,
        ServedTemplateClass.NATIVE_GEMMA3_TURN: has_gemma3,
        ServedTemplateClass.DEGRADED_CHATML: has_chatml,
    }
    present = [family for family, seen in families.items() if seen]
    if len(present) != 1:
        return ServedTemplateClass.UNSUPPORTED
    return present[0]


def stop_markers_for(template_class: ServedTemplateClass) -> frozenset[str]:
    """Return stop-marker substrings for assistant completions under ``template_class``.

    Args:
        template_class: Classified served-template family.

    Returns:
        Marker substrings used to detect premature or stopped completions;
        empty for unsupported families.
    """
    common = frozenset({GEMMA4_TURN_CLOSE, GEMMA4_CHANNEL_CLOSE, GEMMA4_TOOL_RESPONSE})
    if template_class is ServedTemplateClass.DEGRADED_CHATML:
        return common | frozenset({CHATML_IM_END})
    if template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN:
        return common
    if template_class is ServedTemplateClass.NATIVE_GEMMA3_TURN:
        return frozenset({GEMMA3_END_OF_TURN})
    return frozenset()


def label_embeds_control_fragment(label: str) -> bool:
    """Return whether ``label`` embeds a reserved template control fragment.

    Args:
        label: Candidate enum string.

    Returns:
        ``True`` when the label contains a control delimiter.
    """
    return any(fragment in label for fragment in _CONTROL_LABEL_SPLITTERS)
