"""Evaluation tooling for typevet, kept out of the published wheel (#256).

This package is the ``typevet-evals`` uv workspace member. It holds benchmark
runners, dataset loaders, acceptance harnesses and wheel proofs. It imports
``typevet``; ``typevet`` never imports it (import-linter contract "The library
does not import the evals"). The distribution is never uploaded.

Every evaluation family lives in a subpackage or module listed below. The
package itself exports no names; import the family you need.

Attributes:
    __all__ (list[str]): Public names of the package. Empty, because each
        family is imported from its own subpackage or module.

Examples:
    ```python
    import typevet_evals

    assert typevet_evals.__all__ == []
    ```

See Also:
    - [typevet][]: The library that this package evaluates
    - [typevet_evals.throughput][]: Throughput sweep and its workloads
    - [typevet_evals.vllm_acceptance][]: The #170 vLLM acceptance run
    - [typevet_evals.datasets][]: Dataset loaders and the partner data guard
    - [typevet_evals.cli][]: Module-entry commands, such as the eval runner
    - [typevet_evals.wheel_isolated][]: Isolated wheel build and run helpers
    - [typevet_evals.gemma_native_vision_wheel_smoke][]: Factory wheel smoke
    - [typevet_evals.instruction_variant][]: Instruction-variant consumer proof
    - [typevet_evals.outcome_replay_metrics][]: Replay comparison metrics
    - [typevet_evals.psai_vision_consumer][]: PSAI vision consumer proof
    - [typevet_evals.psai_vision_probability_evidence][]: Choice mass evidence
    - [typevet_evals.cord][]: CORD expense smoke and semantic acceptance
    - [typevet_evals.runner][]: Loader eval runner, live gate and reports
    - [typevet_evals.tpjep][]: TPJEP fixture, records and runner
    - [typevet_evals.experiment_identity][]: Receipt identity and snapshots
    - [typevet_evals.calibration][]: Post-hoc calibration of receipt probabilities
    - [typevet_evals.calibration_artifact][]: Calibration map writer
"""

__all__: list[str] = []
