"""vLLM serving-backend adapters (#256).

The package groups the vLLM generation, scoring, image content and HTTP error
modules. The package imports only the three adapters and the scoring framing.
Import the vLLM judgment factory from its own module.

Examples:
    ```python
    from typevet.adapters.outbound.vllm import (
        AsyncVllmGenerationAdapter,
        ChatContentFraming,
        VllmCandidateScoringAdapter,
        VllmGenerationAdapter,
    )
    ```

See Also:
    - [typevet.adapters.outbound.vllm.generation][]: Sync generation
    - [typevet.adapters.outbound.vllm.generation_async][]: Async generation
    - [typevet.adapters.outbound.vllm.scoring][]: Chat logprob scorer
    - [typevet.adapters.outbound.vllm.content][]: Image content blocks
    - [typevet.adapters.outbound.vllm.http_mapping][]: HTTP error mapping
    - [typevet.adapters.outbound.vllm.judgment_factory][]: vLLM judgment factory

Attributes:
    AsyncVllmGenerationAdapter (type): Async vLLM generation with a POST limit.
    ChatContentFraming (type): Plain chat content framing for vLLM scoring.
    VllmCandidateScoringAdapter (type): vLLM chat completions logprob scorer.
    VllmGenerationAdapter (type): vLLM structured-output generation adapter.
"""

from typevet.adapters.outbound.vllm.generation import VllmGenerationAdapter
from typevet.adapters.outbound.vllm.generation_async import AsyncVllmGenerationAdapter
from typevet.adapters.outbound.vllm.scoring import (
    ChatContentFraming,
    VllmCandidateScoringAdapter,
)

__all__ = [
    "AsyncVllmGenerationAdapter",
    "ChatContentFraming",
    "VllmCandidateScoringAdapter",
    "VllmGenerationAdapter",
]
