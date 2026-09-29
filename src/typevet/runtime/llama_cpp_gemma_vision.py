"""Public re-exports for the Gemma native vision factory ([#174][i174]).

Library callers import from ``typevet.runtime``; the implementation lives in
``typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory`` so evaluation harnesses
can compose the same factory without breaking hex layer rules.

Examples:
    ```python
    from typevet.runtime import open_gemma_native_vision_judgment
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: implementation

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
"""

from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
    probe_gemma_native_vision_support,
)

__all__ = [
    "GemmaNativeVisionSession",
    "open_gemma_native_vision_judgment",
    "probe_gemma_native_vision_support",
]
