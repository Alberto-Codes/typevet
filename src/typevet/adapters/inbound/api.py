"""Library entry: call a generation port with a typed request.

Examples:
    ```python
    from typevet.adapters.inbound.api import generate
    from typevet.testing import StaticGenerationFake

    generate(
        StaticGenerationFake({"n": 1}),
        prompt="n",
        schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        },
        model="fake",
    )
    ```

See Also:
    - [typevet.domain.models][]: GenerationRequest
    - [typevet.ports.generation][]: GenerationPort
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.ports.generation import GenerationPort


def generate(
    port: GenerationPort,
    *,
    prompt: str,
    schema: Mapping[str, Any],
    model: str,
) -> GenerationResult:
    """Build a request and invoke the generation port.

    Args:
        port: Outbound adapter that implements ``GenerationPort``.
        prompt: Natural-language instruction.
        schema: JSON Schema object as a mapping.
        model: Backend model id or alias.

    Returns:
        Validated generation result from the port.
    """
    request = GenerationRequest(prompt=prompt, schema=schema, model=model)
    return port.generate(request)
