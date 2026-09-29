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
    - [typevet._version][]: Resolves ``__version__`` from distribution metadata
    - [typevet.domain][]: Request, result and error types
    - [typevet.ports][]: GenerationPort protocol
    - [typevet.adapters.outbound][]: llama.cpp and fake adapters
    - [typevet.runtime][]: Source of the ``decide_categorical`` re-export

Attributes:
    BackendHttpError (type): llama.cpp HTTP error status with body snippet.
    GenerationError (type): Base failure for a generation call.
    TransportError (type): HTTP client failure before a response.
    AsyncGenerationPort (type): Structural protocol for async typed generation.
    GenerationPort (type): Structural protocol for typed generation.
    GenerationRequest (type): Prompt, schema and model ask.
    GenerationResult (type): Validated structured value.
    SchemaValidationError (type): Output failed the requested schema.
    decide_categorical (callable): M1 categorical decision via scoring port;
        re-exported from ``typevet.runtime.categorical``.
    __version__ (str): Installed distribution version; matches
        ``importlib.metadata.version("typevet")`` when the package is on
        ``PYTHONPATH``. Exported in ``__all__``.
"""

from typevet._version import __version__
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    SchemaValidationError,
    TransportError,
)
from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.ports.async_generation import AsyncGenerationPort
from typevet.ports.generation import GenerationPort
from typevet.runtime.categorical import decide_categorical

__all__ = [
    "AsyncGenerationPort",
    "BackendHttpError",
    "GenerationError",
    "GenerationPort",
    "GenerationRequest",
    "GenerationResult",
    "SchemaValidationError",
    "TransportError",
    "__version__",
    "decide_categorical",
]
