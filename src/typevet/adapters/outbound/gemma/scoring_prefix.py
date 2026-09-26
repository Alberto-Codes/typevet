"""Scoring-prefix composition for Gemma judgment (#148, #157, #179).

Text-only scoring uses degraded ChatML. Media scoring uses the served native
turn family: Gemma 3 ``<start_of_turn>`` or Gemma 4 ``<|turn>``.

Examples:
    ```python
    from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix
    ```

See Also:
    - [typevet.domain.field_instructions][]: Model-agnostic field blocks
    - [typevet.runtime.scoring_prefix][]: Runtime re-export
    - [typevet.adapters.outbound.gemma.served_template][]: Family classifier
"""

from __future__ import annotations

from typing import Final

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
)
from typevet.domain.errors import GemmaTemplateError

_NATIVE_TURN_WRAPPERS: Final[dict[ServedTemplateClass, tuple[str, str, str]]] = {
    ServedTemplateClass.NATIVE_GEMMA3_TURN: (
        GEMMA3_START_OF_TURN,
        GEMMA3_END_OF_TURN,
        GEMMA3_MODEL_TURN_HEADER,
    ),
    ServedTemplateClass.NATIVE_GEMMA4_TURN: (
        GEMMA4_TURN_OPEN,
        GEMMA4_TURN_CLOSE,
        GEMMA4_MODEL_TURN_HEADER,
    ),
}


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


def compose_media_scoring_prefix(
    *,
    context: str,
    field_block: str,
    template_class: ServedTemplateClass,
) -> str:
    """Compose a native-turn prefix for image-conditioned scoring.

    The shape matches llama.cpp ``/apply-template`` output for the served
    family, so media markers sit inside the native user turn.

    Args:
        context: User text, including one media marker per image.
        field_block: Rendered field instructions from ``render_field_instructions``.
        template_class: Classified served-template family of the scoring model.

    Returns:
        Prefix string ending with the model turn header of ``template_class``.

    Raises:
        GemmaTemplateError: ``template_class`` is not a native turn family.
    """
    wrappers = _NATIVE_TURN_WRAPPERS.get(template_class)
    if wrappers is None:
        msg = (
            "media scoring prefix needs a native Gemma 3 or Gemma 4 served "
            f"template (class={template_class.value})"
        )
        raise GemmaTemplateError(msg)
    turn_open, turn_close, model_header = wrappers
    return f"{turn_open}user\n{context}\n\n{field_block}{turn_close}\n{model_header}"
