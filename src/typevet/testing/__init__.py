"""Test doubles that stay off adapters.

Examples:
    ```python
    from typevet.testing import StaticGenerationFake

    StaticGenerationFake({"ok": True})
    ```

See Also:
    - [typevet.testing.fakes][]: StaticGenerationFake
    - [typevet.ports][]: GenerationPort

Attributes:
    StaticGenerationFake (type): Fixed-value port double without adapter imports.
"""

from typevet.testing.fakes import StaticGenerationFake

__all__ = ["StaticGenerationFake"]
