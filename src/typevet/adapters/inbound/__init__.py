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
    - [typevet.adapters.inbound.backend_settings][]: ``TYPEVET_BACKEND`` selection
    - [typevet.ports.generation][]: GenerationPort

Attributes:
    generate (function): Build a request and invoke a generation port.
    run_sync (function): Run an async generation coroutine from sync code.
    load_calibration_map (function): Read a calibration map file and check its sha256.
    LlamaSettings (type): llama.cpp connection settings for composition roots.
    load_llama_settings (function): Read ``TYPEVET_LLAMA__*`` from the environment.
    llama_cpp_adapter (function): Build ``LlamaCppGenerationAdapter`` from settings.
    VllmSettings (type): vLLM connection and API gateway settings; ``api_key``
        and ``headers`` are not in ``repr``.
    load_backend (function): Read ``TYPEVET_BACKEND``.
    load_vllm_settings (function): Read ``TYPEVET_VLLM__*`` from the environment.
    vllm_http_client (function): Build the shared vLLM ``httpx.Client``.
    generation_adapter (function): Build the adapter ``TYPEVET_BACKEND`` selects.
    async_vllm_generation_adapter (function): Build the async vLLM adapter.
"""

from typevet.adapters.inbound.api import generate
from typevet.adapters.inbound.backend_settings import (
    VllmSettings,
    async_vllm_generation_adapter,
    generation_adapter,
    load_backend,
    load_vllm_settings,
    vllm_http_client,
)
from typevet.adapters.inbound.calibration_map import load_calibration_map
from typevet.adapters.inbound.helpers import run_sync
from typevet.adapters.inbound.settings import (
    LlamaSettings,
    llama_cpp_adapter,
    load_llama_settings,
)

__all__ = [
    "LlamaSettings",
    "VllmSettings",
    "async_vllm_generation_adapter",
    "generate",
    "generation_adapter",
    "llama_cpp_adapter",
    "load_backend",
    "load_calibration_map",
    "load_llama_settings",
    "load_vllm_settings",
    "run_sync",
    "vllm_http_client",
]
