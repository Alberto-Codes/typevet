"""Pure domain types for typed generation.

Examples:
    ```python
    from typevet.domain import GenerationRequest

    req = GenerationRequest(
        prompt="Classify sentiment.",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        model="gemma-4-31b-24gib-kv11-decoder",
    )
    assert req.model.startswith("gemma")
    ```

See Also:
    - [typevet.domain.models][]: Request and result dataclasses
    - [typevet.domain.errors][]: Generation failures

Attributes:
    GenerationError (type): Base failure for a generation call.
    GenerationRequest (type): Prompt, schema and model ask.
    GenerationResult (type): Validated structured value.
    SchemaValidationError (type): Output failed the requested schema.
"""

from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest, GenerationResult

__all__ = [
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "SchemaValidationError",
]
