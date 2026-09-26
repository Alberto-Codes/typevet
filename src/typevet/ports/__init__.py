"""Ports for typed generation.

Examples:
    ```python
    from typevet.ports import GenerationPort
    from typevet.testing import StaticGenerationFake

    port: GenerationPort = StaticGenerationFake({"ok": True})
    ```

See Also:
    - [typevet.ports.generation][]: GenerationPort protocol
    - [typevet.adapters.outbound][]: Adapters that implement the port

Attributes:
    GenerationPort (type): Structural protocol for typed generation.
"""

from typevet.ports.generation import GenerationPort

__all__ = ["GenerationPort"]
