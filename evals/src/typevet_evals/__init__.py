"""Evaluation tooling for typevet, kept out of the published wheel (#256).

This package is the ``typevet-evals`` uv workspace member. It holds benchmark
runners, dataset loaders, acceptance harnesses and wheel proofs. It imports
``typevet``; ``typevet`` never imports it (import-linter contract "The library
does not import the evals"). The distribution is never uploaded.

The evaluation families move here from ``typevet.evaluation`` in later
children of #256. Until then the package exports no names.

Attributes:
    __all__ (list[str]): Public names of the package. Empty until the
        evaluation families move in.

Examples:
    ```python
    import typevet_evals

    assert typevet_evals.__all__ == []
    ```

See Also:
    - [typevet][]: The library that this package evaluates
    - [typevet_evals.throughput][]: Throughput sweep and its workloads
    - [typevet_evals.vllm_acceptance][]: The #170 vLLM acceptance run
    - [typevet.evaluation][]: Evaluation code that has not moved yet
    - [typevet_evals.cli][]: Module-entry commands, such as the eval runner
    - [typevet_evals.wheel_isolated][]: Isolated wheel build and run helpers
    - [typevet_evals.gemma_native_vision_wheel_smoke][]: Factory wheel smoke
    - [typevet_evals.instruction_variant][]: Instruction-variant consumer proof
    - [typevet_evals.outcome_replay_metrics][]: Replay comparison metrics
    - [typevet_evals.psai_vision_consumer][]: PSAI vision consumer proof
    - [typevet_evals.psai_vision_probability_evidence][]: Choice mass evidence
    - [typevet_evals.cord][]: CORD expense smoke and semantic acceptance
"""

__all__: list[str] = []
