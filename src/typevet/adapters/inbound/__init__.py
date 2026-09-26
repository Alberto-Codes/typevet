"""Inbound adapters (library entry points).

Examples:
    ```python
    from typevet.adapters.inbound import generate
    from typevet.testing import StaticGenerationFake

    result = generate(
        StaticGenerationFake({"ok": True}),
        prompt="hi",
        schema={"type": "object", "additionalProperties": False},
        model="fake",
    )
    ```

See Also:
    - [typevet.adapters.inbound.api][]: ``generate`` helper
    - [typevet.ports.generation][]: GenerationPort

Attributes:
    generate (function): Build a request and invoke a generation port.
"""

from typevet.adapters.inbound.api import generate

__all__ = ["generate"]
