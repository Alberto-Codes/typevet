"""Throughput runs over served judgments: workloads, sweep and ``/metrics`` (#236).

The collections workload, the public-dataset workloads, the concurrency sweep
and the vLLM ``/metrics`` deltas moved here from ``typevet.evaluation`` (#256).
The module names did not change. Module constants stay on their modules; this
package re-exports the classes and functions.

Attributes:
    Snapshot (type): ``/metrics`` sums keyed by sample name and ``le``.
    nearest_rank (function): Nearest-rank quantile of sorted values.
    latency (function): Client seconds as nearest-rank p50, p95 and p99.
    read_metrics (function): Read the ``/metrics`` text once.
    snapshot (function): Parse Prometheus text into a ``Snapshot``.
    histogram_delta (function): Change of one histogram between two reads.
    prefix_hit_rate (function): Prefix-cache hits over prefix-cache queries.
    server_delta (function): Server-side histogram and prefix-cache deltas.
    ScoredRecord (type): A record the runner can send and score.
    Scoring (type): Optional ``run_throughput`` keywords for parity scoring.
    RunOptions (type): Time caps and caller-supplied receipt values.
    level_timing (function): Wall clock and answered-record rate of a level.
    remaining_run_seconds (function): Run budget that an earlier receipt left.
    best_level (function): Level with the highest records/s and at most 1%
        errors.
    run_throughput (function): Run the concurrency sweep once and return the
        receipt mapping.
    CollectionsRecord (type): One mapped collections record.
    Baseline (type): Reference values that one parity check compares against.
    Bin (type): One reliability bin.
    map_record (function): Map one finvet split record without changing it.
    load_records (function): Read and map a jsonl split in file order.
    workload_paths (function): Resolve the split file and the seed file from
        the environment.
    load_questions (function): Read the outcome seed into typed questions.
    judge_record (function): Send one record with both questions in one call.
    reliability (function): Bin probabilities against labels.
    ece (function): Expected calibration error as finvet computes it.
    parity (function): Build the quality parity record for answered records.
    PublicRecord (type): One public record: the text sent and its label.
    PublicWorkload (type): Records, questions, scored Noul name and baseline
        of one set.
    validate_questions (function): Reject a Choice with more options than the
        native limit.
    banking77_workload (function): Build the Banking77 workload.
    difraud_workload (function): Build the DIFrauD workload.
    missing_data (function): Why the data directory cannot feed a run, or
        ``None``.
    fetch_public_data (function): Download the Banking77 and DIFrauD SMS test
        files.
    load_public_workloads (function): Build the three #236 sets from the
        fetched files, offline.

Examples:
    ```python
    from typevet_evals.throughput import load_public_workloads, run_throughput
    ```

See Also:
    - [typevet_evals.throughput.collections_metrics][]: ``/metrics`` deltas
    - [typevet_evals.throughput.collections_throughput][]: the sweep runner
    - [typevet_evals.throughput.collections_workload][]: collections workload
    - [typevet_evals.throughput.public_workload][]: public-dataset workloads
    - [typevet_evals.vllm_acceptance][]: call caps and the receipt writer
"""

from typevet_evals.throughput.collections_metrics import (
    Snapshot,
    histogram_delta,
    latency,
    nearest_rank,
    prefix_hit_rate,
    read_metrics,
    server_delta,
    snapshot,
)
from typevet_evals.throughput.collections_throughput import (
    RunOptions,
    ScoredRecord,
    Scoring,
    best_level,
    level_timing,
    remaining_run_seconds,
    run_throughput,
)
from typevet_evals.throughput.collections_workload import (
    Baseline,
    Bin,
    CollectionsRecord,
    ece,
    judge_record,
    load_questions,
    load_records,
    map_record,
    parity,
    reliability,
    workload_paths,
)
from typevet_evals.throughput.public_workload import (
    PublicRecord,
    PublicWorkload,
    banking77_workload,
    difraud_workload,
    fetch_public_data,
    load_public_workloads,
    missing_data,
    validate_questions,
)

__all__ = [
    "Baseline",
    "Bin",
    "CollectionsRecord",
    "PublicRecord",
    "PublicWorkload",
    "RunOptions",
    "ScoredRecord",
    "Scoring",
    "Snapshot",
    "banking77_workload",
    "best_level",
    "difraud_workload",
    "ece",
    "fetch_public_data",
    "histogram_delta",
    "judge_record",
    "latency",
    "level_timing",
    "load_public_workloads",
    "load_questions",
    "load_records",
    "map_record",
    "missing_data",
    "nearest_rank",
    "parity",
    "prefix_hit_rate",
    "read_metrics",
    "reliability",
    "remaining_run_seconds",
    "run_throughput",
    "server_delta",
    "snapshot",
    "validate_questions",
    "workload_paths",
]
