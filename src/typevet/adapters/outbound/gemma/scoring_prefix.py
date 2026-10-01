"""Scoring-prefix composition for Gemma judgment (#148, #157, #179, #187).

Text-only scoring uses degraded ChatML. Media scoring uses the served native
turn family: Gemma 3 ``<start_of_turn>`` or Gemma 4 ``<|turn>`` with
no-thinking prefill after the model header. A caller framing whose last turn
is a Gemma 4 model turn must end with that prefill (#354).

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
    GEMMA4_NO_THINKING_PREFILL,
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
_TURN_OPENERS: Final[tuple[str, ...]] = (
    CHATML_IM_START,
    GEMMA3_START_OF_TURN,
    GEMMA4_TURN_OPEN,
)


def require_no_thinking_prefill(
    prefix: str, *, field_block: str, framing_class: str
) -> None:
    """Refuse a rendered Gemma 4 prefix that lacks the no-thinking prefill.

    The check reads the rendered text, not the framing object. It reads only
    the tail after the last copy of ``field_block``. typevet renders that
    block; the state text before it is caller data, so the check never reads
    it. The tail is on the Gemma 4 path when its last turn opener starts the
    Gemma 4 model header. That family declares ``GEMMA4_NO_THINKING_PREFILL``;
    the tail must end with it. Other families declare no prefill, so this
    check allows them. A prefix that does not hold ``field_block`` has no
    framing-owned tail, so the check allows it.

    Args:
        prefix: Rendered scoring prefix from one framing.
        field_block: Field instructions typevet gave the framing.
        framing_class: Framing class name; the error names it.

    Raises:
        GemmaTemplateError: The tail's last turn is a Gemma 4 model turn and
            the prefix does not end with the no-thinking prefill. The message
            holds no prompt text.
    """
    block_at = prefix.rfind(field_block)
    if block_at < 0:
        return
    tail = prefix[block_at + len(field_block) :]
    last_turn = max(tail.rfind(opener) for opener in _TURN_OPENERS)
    if last_turn < 0 or not tail.startswith(GEMMA4_MODEL_TURN_HEADER, last_turn):
        return
    if tail.endswith(GEMMA4_NO_THINKING_PREFILL):
        return
    msg = (
        f"framing {framing_class} renders a Gemma 4 model turn without the "
        "no-thinking prefill"
    )
    raise GemmaTemplateError(msg)


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
        Prefix string ending at the answer boundary for ``template_class``.
        Gemma 4 includes the no-thinking thought-channel prefill after the model
        header, matching ``/apply-template`` with ``enable_thinking=false``.

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
    prefix = f"{turn_open}user\n{context}\n\n{field_block}{turn_close}\n{model_header}"
    if template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN:
        return f"{prefix}{GEMMA4_NO_THINKING_PREFILL}"
    return prefix
