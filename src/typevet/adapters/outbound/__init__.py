"""Outbound adapter package.

Examples:
    ```python
    from typevet.adapters.outbound import (
        FakeGenerationAdapter,
        LlamaCppGenerationAdapter,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: Local llama.cpp router adapter
    - [typevet.adapters.outbound.fake][]: Offline validating fake

Attributes:
    FakeGenerationAdapter (type): Offline adapter that validates a fixed value.
    LlamaCppGenerationAdapter (type): OpenAI-compat llama.cpp adapter.
"""

from typevet.adapters.outbound.fake import FakeGenerationAdapter
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter

__all__ = ["FakeGenerationAdapter", "LlamaCppGenerationAdapter"]
