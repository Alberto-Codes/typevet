"""Thin orchestration facades over domain and ports (#148).

Examples:
    ```python
    from typevet.runtime import decide_categorical, judge_with_scoring
    ```

See Also:
    - [typevet.runtime.categorical][]: Categorical decision facade
    - [typevet.runtime.judgment][]: Scoring-backed judgment facade

Attributes:
    decide_categorical (callable): M1 categorical decision via scoring port.
    judge_with_scoring (callable): One-shot scoring-backed judgment helper.
    ScoringJudgmentAdapter (type): Sync ``JudgmentPort`` over ``CandidateScoringPort``.
    compose_scoring_prefix (callable): Degraded ChatML scoring prefix composition.
    GemmaNativeVisionSession (type): Live session from ``open_gemma_native_vision_judgment``.
    open_gemma_native_vision_judgment (callable): Gemma native vision factory context manager.
    probe_gemma_native_vision_support (callable): Router probe without a long-lived port.
    VllmJudgmentSession (type): Session from ``open_vllm_judgment``.
    open_vllm_judgment (callable): vLLM chat-completions judgment context manager.
    vllm_tokenize (callable): vLLM ``/tokenize`` hook factory.
"""

from typevet.runtime.categorical import decide_categorical
from typevet.runtime.judgment import ScoringJudgmentAdapter, judge_with_scoring
from typevet.runtime.llama_cpp_gemma_vision import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
    probe_gemma_native_vision_support,
)
from typevet.runtime.scoring_prefix import compose_scoring_prefix
from typevet.runtime.vllm_judgment import (
    VllmJudgmentSession,
    open_vllm_judgment,
    vllm_tokenize,
)

__all__ = [
    "GemmaNativeVisionSession",
    "ScoringJudgmentAdapter",
    "VllmJudgmentSession",
    "compose_scoring_prefix",
    "decide_categorical",
    "judge_with_scoring",
    "open_gemma_native_vision_judgment",
    "open_vllm_judgment",
    "probe_gemma_native_vision_support",
    "vllm_tokenize",
]
