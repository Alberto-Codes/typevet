"""ChatML thinking framing with an empty think block (#443).

A ChatML thinking model, such as Qwen3.8, starts its answer with a
``<think>`` block. This framing ends the prefix with one known empty think
block, so the first scored token is a menu option and not ``<think>``.

Examples:
    ```python
    from typevet.adapters.outbound.gemma import ChatMLThinkingFraming

    ChatMLThinkingFraming().compose_prefix(user_text="text", media=())
    ```

See Also:
    - [typevet.adapters.outbound.gemma.scoring_prefix][]: Prefill check
    - [typevet.ports.framing][]: Framing port
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_EMPTY_THINK_PREFILLS,
    CHATML_IM_END,
    CHATML_IM_START,
)
from typevet.domain.errors import JudgmentValidationError

if TYPE_CHECKING:
    from typevet.domain.media import ImageInput


class ChatMLThinkingFraming:
    """Compose a text-only ChatML prefix that ends with an empty think block.

    Attributes:
        prefill (str): Empty think block after the assistant header; one of
            ``CHATML_EMPTY_THINK_PREFILLS``.

    Examples:
        ```python
        framing = ChatMLThinkingFraming(prefill=CHATML_EMPTY_THINK_PREFILLS[1])
        framing.compose_prefix(user_text="text", media=())
        ```
    """

    def __init__(self, *, prefill: str = CHATML_EMPTY_THINK_PREFILLS[0]) -> None:
        """Keep one known empty think block.

        Args:
            prefill: Empty think block to put after the assistant header.

        Raises:
            ValueError: ``prefill`` is not in ``CHATML_EMPTY_THINK_PREFILLS``.
        """
        if prefill not in CHATML_EMPTY_THINK_PREFILLS:
            msg = "prefill must be one of CHATML_EMPTY_THINK_PREFILLS"
            raise ValueError(msg)
        self.prefill = prefill

    def compose_prefix(
        self,
        *,
        user_text: str,
        media: tuple[ImageInput, ...],
    ) -> str:
        """Return one ChatML user turn, the assistant header and the prefill.

        Args:
            user_text: User text from the ``context_template`` text part.
            media: Images for the prefix; this framing accepts none.

        Returns:
            Prefix that ends with the assistant header and ``prefill``.

        Raises:
            JudgmentValidationError: ``media`` is not empty.
        """
        if media:
            msg = "ChatML thinking framing is text-only and refuses images"
            raise JudgmentValidationError(msg)
        return (
            f"{CHATML_IM_START}user\n{user_text}{CHATML_IM_END}\n"
            f"{CHATML_ASSISTANT_HEADER}{self.prefill}"
        )
