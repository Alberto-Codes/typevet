"""Ports for typed generation.

Examples:
    ```python
    from typevet.ports import GenerationPort
    from typevet.testing import StaticGenerationFake

    port: GenerationPort = StaticGenerationFake({"ok": True})
    ```

See Also:
    - [typevet.ports.generation][]: GenerationPort protocol
    - [typevet.ports.async_generation][]: AsyncGenerationPort protocol
    - [typevet.adapters.outbound][]: Adapters that implement the port

Attributes:
    AsyncGenerationPort (type): Structural protocol for async typed generation.
    GenerationPort (type): Structural protocol for typed generation.
    JudgmentPort (type): Structural protocol for System One-shaped judgment.
"""

from typevet.ports.async_generation import AsyncGenerationPort
from typevet.ports.generation import GenerationPort
from typevet.ports.judgment import JudgmentPort

__all__ = ["AsyncGenerationPort", "GenerationPort", "JudgmentPort"]
