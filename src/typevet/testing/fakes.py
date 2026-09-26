"""Public offline fakes built from the domain alone.

Examples:
    ```python
    from typevet.testing.fakes import StaticGenerationFake
    from typevet.domain.models import GenerationRequest

    fake = StaticGenerationFake({"ok": True})
    result = fake.generate(
        GenerationRequest(
            prompt="x",
            schema={"type": "object", "additionalProperties": False},
            model="fake",
        )
    )
    assert result.value["ok"] is True
    ```

See Also:
    - [typevet.adapters.outbound.fake][]: Validating fake used in contract tests
    - [typevet.domain.models][]: Request and result types
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.models import GenerationRequest, GenerationResult


class StaticGenerationFake:
    """Return a fixed mapping without importing adapters.

    Callers that need schema validation should use
    ``typevet.adapters.outbound.FakeGenerationAdapter`` in contract tests.
    This fake is for inbound unit tests that only need a port double.

    Attributes:
        _value (dict[str, Any]): Stored mapping returned from generate.

    Examples:
        ```python
        from typevet.testing.fakes import StaticGenerationFake

        StaticGenerationFake({"a": 1})
        ```
    """

    def __init__(self, value: Mapping[str, Any]) -> None:
        """Store the fixed return value.

        Args:
            value: Mapping returned from every ``generate`` call.
        """
        self._value = dict(value)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the fixed value wrapped as a result.

        Args:
            request: Incoming request (model is copied onto the result).

        Returns:
            GenerationResult with the stored mapping.
        """
        return GenerationResult(value=self._value, model=request.model)
