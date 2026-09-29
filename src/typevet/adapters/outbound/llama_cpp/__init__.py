"""llama.cpp serving-backend adapters (#256).

The package groups the llama.cpp generation, scoring, media and HTTP error
modules. The package imports only the three adapters. Import the Gemma native
vision factory from its own module.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp import (
        AsyncLlamaCppGenerationAdapter,
        LlamaCppCandidateScoringAdapter,
        LlamaCppGenerationAdapter,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.generation][]: Sync generation
    - [typevet.adapters.outbound.llama_cpp.generation_async][]: Async generation
    - [typevet.adapters.outbound.llama_cpp.scoring][]: ``/completion`` scorer
    - [typevet.adapters.outbound.llama_cpp.multimodal][]: Media probe and shaping
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: HTTP error mapping
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Gemma
      native vision judgment factory

Attributes:
    AsyncLlamaCppGenerationAdapter (type): Async OpenAI-compat llama.cpp adapter.
    LlamaCppCandidateScoringAdapter (type): Pre-sampling ``/completion`` scorer.
    LlamaCppGenerationAdapter (type): OpenAI-compat llama.cpp adapter.
"""

from typevet.adapters.outbound.llama_cpp.generation import LlamaCppGenerationAdapter
from typevet.adapters.outbound.llama_cpp.generation_async import (
    AsyncLlamaCppGenerationAdapter,
)
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter

__all__ = [
    "AsyncLlamaCppGenerationAdapter",
    "LlamaCppCandidateScoringAdapter",
    "LlamaCppGenerationAdapter",
]
