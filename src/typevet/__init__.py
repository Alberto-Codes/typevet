"""typevet: type-safe structured generation under hexagonal architecture.

Examples:
    ```python
    from typevet import GenerationRequest, GenerationResult
    from typevet.adapters.outbound import FakeGenerationAdapter

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    port = FakeGenerationAdapter(value={"ok": True})
    result = port.generate(
        GenerationRequest(prompt="Say ok.", schema=schema, model="fake")
    )
    assert result.value["ok"] is True
    ```

See Also:
    - [typevet.domain][]: Request, result and error types
    - [typevet.ports][]: GenerationPort protocol
    - [typevet.adapters.outbound][]: llama.cpp and fake adapters

Attributes:
    GenerationError (type): Base failure for a generation call.
    GenerationPort (type): Structural protocol for typed generation.
    GenerationRequest (type): Prompt, schema and model ask.
    GenerationResult (type): Validated structured value.
    SchemaValidationError (type): Output failed the requested schema.
    __version__ (str): Package version string.
"""

from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.ports.generation import GenerationPort

__all__ = [
    "GenerationError",
    "GenerationPort",
    "GenerationRequest",
    "GenerationResult",
    "SchemaValidationError",
]

__version__ = "0.1.0"
