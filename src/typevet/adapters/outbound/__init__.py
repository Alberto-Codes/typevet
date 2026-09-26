"""Outbound adapter package.

Examples:
    ```python
    from typevet.adapters.outbound import (
        AsyncFakeGenerationAdapter,
        AsyncLlamaCppGenerationAdapter,
        FakeGenerationAdapter,
        LlamaCppCandidateScoringAdapter,
        LlamaCppGenerationAdapter,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: Local llama.cpp router adapter
    - [typevet.adapters.outbound.fake][]: Offline validating fake

Attributes:
    AsyncFakeGenerationAdapter (type): Offline async validating fake.
    AsyncLlamaCppGenerationAdapter (type): Async OpenAI-compat llama.cpp adapter.
    FakeGenerationAdapter (type): Offline adapter that validates a fixed value.
    LlamaCppCandidateScoringAdapter (type): Pre-sampling ``/completion`` scorer.
    LlamaCppGenerationAdapter (type): OpenAI-compat llama.cpp adapter.
"""

from typevet.adapters.outbound.async_fake import AsyncFakeGenerationAdapter
from typevet.adapters.outbound.async_llama_cpp import AsyncLlamaCppGenerationAdapter
from typevet.adapters.outbound.fake import FakeGenerationAdapter
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter

__all__ = [
    "AsyncFakeGenerationAdapter",
    "AsyncLlamaCppGenerationAdapter",
    "FakeGenerationAdapter",
    "LlamaCppCandidateScoringAdapter",
    "LlamaCppGenerationAdapter",
]
