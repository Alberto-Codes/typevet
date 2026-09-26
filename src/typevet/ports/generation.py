"""Generation port: schema in, validated value out.

Examples:
    ```python
    from typevet.ports.generation import GenerationPort
    from typevet.testing import StaticGenerationFake


    def use(port: GenerationPort) -> None:
        port.generate  # structural check
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp implementation
    - [typevet.domain.models][]: Request and result types
"""

from __future__ import annotations

from typing import Protocol

from typevet.domain.models import GenerationRequest, GenerationResult


class GenerationPort(Protocol):
    """Structural protocol for typed structured generation.

    Examples:
        ```python
        from typevet.ports.generation import GenerationPort
        from typevet.testing import StaticGenerationFake

        port: GenerationPort = StaticGenerationFake({"a": 1})
        ```
    """

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a value that validates against ``request.schema``.

        Args:
            request: Prompt, JSON Schema mapping and model id.

        Returns:
            A validated ``GenerationResult``.

        Raises:
            typevet.domain.errors.TransportError: HTTP client failure (no response).
            typevet.domain.errors.BackendHttpError: llama.cpp HTTP status 400+.
            typevet.domain.errors.GenerationError: Other parse or shape failure.
            typevet.domain.errors.SchemaValidationError: Output fails the schema
                (fail-fast; not retried in-repo).
            typevet.domain.decisions.SchemaError: Not raised here; raised by
                ``compile_json_schema`` when the schema cannot be compiled.
        """
        ...
