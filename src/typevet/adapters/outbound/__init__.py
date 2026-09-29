"""Outbound adapter package.

Examples:
    ```python
    from typevet.adapters.outbound import (
        AsyncFakeGenerationAdapter,
        AsyncLlamaCppGenerationAdapter,
        AsyncVllmGenerationAdapter,
        ChatContentFraming,
        FakeGenerationAdapter,
        LlamaCppCandidateScoringAdapter,
        LlamaCppGenerationAdapter,
        VllmCandidateScoringAdapter,
        VllmGenerationAdapter,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp router adapters
    - [typevet.adapters.outbound.fake][]: Offline validating fake
    - [typevet.adapters.outbound.vllm][]: vLLM generation and scoring adapters
    - [typevet.adapters.outbound.chat_completion][]: Shared content and schema checks
    - [typevet.adapters.outbound.http_errors][]: Shared HTTP error limits

Attributes:
    AsyncFakeGenerationAdapter (type): Offline async validating fake.
    AsyncLlamaCppGenerationAdapter (type): Async OpenAI-compat llama.cpp adapter.
    AsyncVllmGenerationAdapter (type): Async vLLM generation with a POST limit.
    ChatContentFraming (type): Plain chat content framing for vLLM scoring.
    FakeGenerationAdapter (type): Offline adapter that validates a fixed value.
    LlamaCppCandidateScoringAdapter (type): Pre-sampling ``/completion`` scorer.
    LlamaCppGenerationAdapter (type): OpenAI-compat llama.cpp adapter.
    VllmCandidateScoringAdapter (type): vLLM chat completions logprob scorer.
    VllmGenerationAdapter (type): vLLM structured-output generation adapter.
"""

from typevet.adapters.outbound.async_fake import AsyncFakeGenerationAdapter
from typevet.adapters.outbound.fake import FakeGenerationAdapter
from typevet.adapters.outbound.llama_cpp import (
    AsyncLlamaCppGenerationAdapter,
    LlamaCppCandidateScoringAdapter,
    LlamaCppGenerationAdapter,
)
from typevet.adapters.outbound.vllm import (
    AsyncVllmGenerationAdapter,
    ChatContentFraming,
    VllmCandidateScoringAdapter,
    VllmGenerationAdapter,
)

__all__ = [
    "AsyncFakeGenerationAdapter",
    "AsyncLlamaCppGenerationAdapter",
    "AsyncVllmGenerationAdapter",
    "ChatContentFraming",
    "FakeGenerationAdapter",
    "LlamaCppCandidateScoringAdapter",
    "LlamaCppGenerationAdapter",
    "VllmCandidateScoringAdapter",
    "VllmGenerationAdapter",
]
