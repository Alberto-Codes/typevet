"""Public re-exports for the vLLM judgment factory ([#169][i169]).

Library callers import from ``typevet.runtime``; the implementation lives in
``typevet.adapters.outbound.vllm.judgment_factory`` so evaluation harnesses
can compose the same factory without breaking hex layer rules.

Examples:
    ```python
    from typevet.runtime import open_vllm_judgment
    ```

See Also:
    - [typevet.adapters.outbound.vllm.judgment_factory][]: implementation

[i169]: https://github.com/Alberto-Codes/typevet/issues/169
"""

from typevet.adapters.outbound.vllm.judgment_factory import (
    VllmJudgmentSession,
    open_vllm_judgment,
    vllm_tokenize,
)

__all__ = [
    "VllmJudgmentSession",
    "open_vllm_judgment",
    "vllm_tokenize",
]
