"""Model framing port: wrap judgment text in one model's turn markers (#174).

A framing owns the chat-turn markers for one model family. It does not own
the transport payload or the user text. ``ScoringJudgmentAdapter`` renders the
user text with the ``context_template`` text part, passes it to an optional
framing and sends the composed prefix to any ``CandidateScoringPort`` (#373).

A framing for a thinking model on llama.cpp must end the prefix with the
no-thinking prefill. For Gemma 4 that prefill is ``GEMMA4_NO_THINKING_PREFILL``
in ``typevet.adapters.outbound.gemma.served_template``. The scoring adapter
checks the rendered prefix before the scoring call and refuses a Gemma 4 model
turn without it (#354). Without the prefill, the #207 receipt put at least 0.99999 of
the mass off the menu, and the answers changed ([#235][i235]). On vLLM the
scoring adapter sends ``chat_template_kwargs: {"enable_thinking": False}``, so
a vLLM framing needs no prefill text.

Examples:
    ```python
    from typevet.ports.framing import ModelFramingPort


    def use(framing: ModelFramingPort) -> str:
        return framing.compose_prefix(user_text="text", media=())
    ```

See Also:
    - [typevet.ports.scoring][]: Transport that receives the composed prefix
    - [typevet.domain.media][]: ``ImageInput`` and the media marker

[i235]: https://github.com/Alberto-Codes/typevet/issues/235
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typevet.domain.media import ImageInput


class ModelFramingPort(Protocol):
    """Structural protocol for model-specific scoring-prefix framing.

    Examples:
        ```python
        from typevet.ports.framing import ModelFramingPort

        framing: ModelFramingPort
        _ = framing.compose_prefix
        ```
    """

    def compose_prefix(
        self,
        *,
        user_text: str,
        media: tuple[ImageInput, ...],
    ) -> str:
        """Wrap the rendered user text in this model's turn markers.

        Args:
            user_text: User text that the ``context_template`` text part
                rendered from the state context and the field block. It holds
                one media marker per image.
            media: Images the prefix marks, in order.

        Returns:
            Scoring prefix ending at the answer boundary. The prefix must keep
            one media marker per entry in ``media``. For a thinking model on
            llama.cpp, the prefix must end with the no-thinking prefill.
        """
        ...
