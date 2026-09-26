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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse


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
    ) -> JudgmentResponse:
        """Evaluate ``state`` against named questions.

        Args:
            state: Content under evaluation (text, JSON object, or array).
            questions: Question names to typed questions or raw wire dictionaries.
            model: Backend model id or alias.

        Returns:
            Typed ``JudgmentResponse`` with one answer per question.

        Raises:
            typevet.domain.errors.JudgmentError: When the call fails before answers
                exist. Concrete subclasses depend on the adapter.
        """
        ...
