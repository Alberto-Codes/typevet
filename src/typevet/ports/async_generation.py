"""Async generation port: schema in, validated value out.

Examples:
    ```python
    from typevet.ports.async_generation import AsyncGenerationPort
    from typevet.adapters.outbound import AsyncFakeGenerationAdapter

    port: AsyncGenerationPort = AsyncFakeGenerationAdapter(value={"a": 1})
    ```

See Also:
    - [typevet.ports.generation][]: Sync GenerationPort
    - [typevet.adapters.outbound.async_llama_cpp][]: llama.cpp implementation
    - [typevet.domain.models][]: Request and result types
"""

from __future__ import annotations

from typing import Protocol

from typevet.domain.models import GenerationRequest, GenerationResult


class AsyncGenerationPort(Protocol):
    """Structural protocol for async typed structured generation.

    Examples:
        ```python
        from typevet.ports.async_generation import AsyncGenerationPort
        from typevet.adapters.outbound import AsyncFakeGenerationAdapter

        port: AsyncGenerationPort = AsyncFakeGenerationAdapter(value={"a": 1})
        ```
    """

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a value that validates against ``request.schema``.

        Args:
            request: Prompt, JSON Schema mapping and model id.

        Returns:
            A validated ``GenerationResult``.

        Raises:
            typevet.domain.errors.GenerationError: On transport or parse failure.
            typevet.domain.errors.SchemaValidationError: When output fails the schema.
        """
        ...
