"""Model framing port: wrap judgment text in one model's turn markers (#174).

A framing owns the chat-turn markers for one model family. It does not own
the transport payload. ``ScoringJudgmentAdapter`` takes an optional framing
and sends the composed prefix to any ``CandidateScoringPort``.

Examples:
    ```python
    from typevet.ports.framing import ModelFramingPort


    def use(framing: ModelFramingPort) -> str:
        return framing.compose_prefix(context="c", field_block="f", media=())
    ```

See Also:
    - [typevet.ports.scoring][]: Transport that receives the composed prefix
    - [typevet.domain.media][]: ``ImageInput`` and the media marker
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
        context: str,
        field_block: str,
        media: tuple[ImageInput, ...],
    ) -> str:
        """Wrap context and field block in this model's turn markers.

        Args:
            context: Rendered state context; holds one media marker per image.
            field_block: Rendered field instructions.
            media: Images the prefix marks, in order.

        Returns:
            Scoring prefix ending at the answer boundary. The prefix must keep
            one media marker per entry in ``media``.
        """
        ...
