"""Judgment port: state and questions in, typed answers out.

The shape aligns with judgevet ``SystemOnePort`` vocabulary without importing
judgevet. Live llama.cpp scoring adapters are out of scope for this protocol
module.

Examples:
    ```python
    from typevet.ports.judgment import JudgmentPort


    def use(port: JudgmentPort) -> None:
        port.judge  # structural check
    ```

See Also:
    - [typevet.domain.judgment_response][]: Response container
    - [typevet.domain.judgment_questions][]: Question types
    - [typevet.domain.media][]: ``ImageInput`` and the media marker
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput


class JudgmentPort(Protocol):
    """Structural protocol for System One-shaped typed judgment.

    Examples:
        ```python
        from typevet.ports.judgment import JudgmentPort

        port: JudgmentPort
        _ = port.judge
        ```
    """

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Evaluate ``state`` against named questions.

        ``None`` or an empty ``media`` tuple is the text path and behaves as it
        did before images existed. A non-empty tuple conditions every scored
        field on the same images.

        A set ``off_option_threshold`` flags each answer whose off-option mass
        is above it, and does not raise for that mass. The answer receipt in
        ``JudgmentResponse.off_option`` holds the mass, the threshold and the
        flag. ``None`` turns the guard off. A wrapper forwards the value to
        the port it wraps.

        Args:
            state: Content under evaluation (text, JSON object, or array).
            questions: Question names to typed questions or raw wire dictionaries.
            model: Backend model id or alias.
            media: Images to condition every scored field on, in order.
            off_option_threshold: Off-option mass limit in ``[0, 1]``, or
                ``None`` (the default) to turn the guard off.

        Returns:
            Typed ``JudgmentResponse`` with one answer per question.

        Raises:
            typevet.domain.errors.JudgmentError: When the call fails before answers
                exist, or the threshold is not ``None`` or in ``[0, 1]``.
                Concrete subclasses depend on the adapter.
        """
        ...
