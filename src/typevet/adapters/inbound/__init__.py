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
    - [typevet.adapters.inbound.helpers][]: ``run_sync`` helper
    - [typevet.ports.generation][]: GenerationPort

Attributes:
    generate (function): Build a request and invoke a generation port.
    run_sync (function): Run an async generation coroutine from sync code.
"""

from typevet.adapters.inbound.api import generate
from typevet.adapters.inbound.helpers import run_sync

__all__ = ["generate", "run_sync"]
