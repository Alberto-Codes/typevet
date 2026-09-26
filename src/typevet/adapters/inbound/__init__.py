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
    - [typevet.adapters.inbound.settings][]: ``TYPEVET_LLAMA__*`` composition root
    - [typevet.ports.generation][]: GenerationPort

Attributes:
    generate (function): Build a request and invoke a generation port.
    run_sync (function): Run an async generation coroutine from sync code.
    LlamaSettings (type): llama.cpp connection settings for composition roots.
    load_llama_settings (function): Read ``TYPEVET_LLAMA__*`` from the environment.
    llama_cpp_adapter (function): Build ``LlamaCppGenerationAdapter`` from settings.
"""

from typevet.adapters.inbound.api import generate
from typevet.adapters.inbound.helpers import run_sync
from typevet.adapters.inbound.settings import (
    LlamaSettings,
    llama_cpp_adapter,
    load_llama_settings,
)

__all__ = [
    "LlamaSettings",
    "generate",
    "llama_cpp_adapter",
    "load_llama_settings",
    "run_sync",
]
