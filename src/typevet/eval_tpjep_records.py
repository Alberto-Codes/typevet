"""Compatibility shim for TPJEP attempt records (#147).

Prefer importing from ``typevet.evaluation.tpjep.records`` in new code.

Examples:
    ```python
    from typevet.eval_tpjep_records import iter_records_jsonl
    ```

See Also:
    - [typevet.evaluation.tpjep.records][]: New home for this module
"""

from typevet.evaluation.tpjep.records import (
    TPJEP_PROTOCOL_V0,
    RecordJson,
    TpjepAttemptRecord,
    TpjepOutcome,
    TpjepRunSummary,
    iter_records_jsonl,
    record_from_dict,
    record_to_dict,
    records_to_jsonl,
    summarize_tpjep_records,
)

__all__ = [
    "TPJEP_PROTOCOL_V0",
    "RecordJson",
    "TpjepAttemptRecord",
    "TpjepOutcome",
    "TpjepRunSummary",
    "iter_records_jsonl",
    "record_from_dict",
    "record_to_dict",
    "records_to_jsonl",
    "summarize_tpjep_records",
]
