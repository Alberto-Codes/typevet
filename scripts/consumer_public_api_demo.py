"""Bounded public-API demo: replay compare without model calls ([#132][#177]).

Examples:
    ```bash
    uv run python scripts/consumer_public_api_demo.py
    ```

See Also:
    - [typevet.evaluation.outcome_replay_metrics][]: offline compare API
"""

from __future__ import annotations

import json
from pathlib import Path

from typevet.evaluation.outcome_replay_metrics import (
    SavedPromptOutcome,
    compare_matched_prompt_outcomes,
)


def main() -> int:
    """Write a replayable JSON record comparing two frozen instruction arms.

    Returns:
        Process exit code (0 on success).
    """
    gold = {"c1": "yes", "c2": "no"}
    seed = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.35, "yes": 0.65}),
        "c2": SavedPromptOutcome(probabilities={"no": 0.55, "yes": 0.45}),
    }
    candidate = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.20, "yes": 0.80}),
        "c2": SavedPromptOutcome(probabilities={"no": 0.70, "yes": 0.30}),
    }
    report = compare_matched_prompt_outcomes(
        gold, seed, candidate, labels=("no", "yes")
    )
    record = {
        "instruction_variants": ["seed-v1", "candidate-v2"],
        "gold": gold,
        "report": report,
    }
    out = Path("scratchpad/consumer-usability/public-api-demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(out),
                "shared_valid_case_ids": report["shared_valid_case_ids"],
                "replay_report_schema": report["replay_report_schema"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
