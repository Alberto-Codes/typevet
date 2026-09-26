"""Request and result values for typed generation.

Examples:
    ```python
    from typevet.domain.models import GenerationRequest, GenerationResult

    req = GenerationRequest(
        prompt="hi",
        schema={"type": "object", "additionalProperties": False},
        model="m",
    )
    result = GenerationResult(value={"ok": True}, model=req.model)
    assert result.model == "m"
    ```

See Also:
    - [typevet.domain.errors][]: Failures raised around these values
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One typed generation ask.

    Attributes:
        prompt (str): Natural-language instruction for the model.
        schema (Mapping[str, Any]): JSON Schema object as a mapping.
        model (str): Backend model id or alias.

    Examples:
        ```python
        from typevet.domain.models import GenerationRequest

        GenerationRequest(
            prompt="Return JSON.",
            schema={"type": "object", "additionalProperties": False},
            model="fake",
        )
        ```
    """

    prompt: str
    schema: Mapping[str, Any]
    model: str

    def __post_init__(self) -> None:
        """Reject empty prompt or model and a non-object schema root.

        Raises:
            ValueError: When prompt or model is blank, or schema type is not object.
            TypeError: When schema is not a mapping.
        """
        if not self.prompt.strip():
            msg = "prompt must be non-empty"
            raise ValueError(msg)
        if not self.model.strip():
            msg = "model must be non-empty"
            raise ValueError(msg)
        if not isinstance(self.schema, Mapping):
            msg = "schema must be a mapping"
            raise TypeError(msg)
        schema_type = self.schema.get("type")
        if schema_type is not None and schema_type != "object":
            msg = "schema root type must be object when set"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A value that validated against the request schema.

    Attributes:
        value (Mapping[str, Any]): Validated JSON-compatible mapping.
        model (str): Model id that produced the value.
        raw_text (str | None): Optional raw model text before parse.

    Examples:
        ```python
        from typevet.domain.models import GenerationResult

        GenerationResult(value={"ok": True}, model="fake")
        ```
    """

    value: Mapping[str, Any]
    model: str
    raw_text: str | None = None
