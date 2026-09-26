"""Frozen pins for the instruction-variant consumer slice ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.instruction_variant_consumer_protocol import (
        FROZEN_VARIANT_SCORING_REQUESTS,
        plan_instruction_variant_calls,
    )

    plan = plan_instruction_variant_calls()
    assert plan.scoring_requests <= FROZEN_VARIANT_SCORING_REQUESTS
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_proof][]: harness runner
    - [docs/maintainers/consumer-instruction-variant-protocol.md][]: maintainer table

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from dataclasses import dataclass

from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallCounts

INSTRUCTION_VARIANT_PROTOCOL_REVISION = 1

FROZEN_VARIANT_CASE_UIDS: frozenset[str] = frozenset(
    {
        "cmcc8u6yc00v91p1yw2eruz95",
        "cmcc8u6yd00wv1p1yy8guorre",
    }
)

DEFAULT_SEED_INSTRUCTION = "Is the website in this screenshot Fox News?"
DEFAULT_CANDIDATE_INSTRUCTION = (
    "Does the attached screenshot show the Fox News website chrome (logo and header)?"
)

FROZEN_VARIANT_JUDGMENT_CALLS = 4
FROZEN_VARIANT_SCORING_REQUESTS = 8
FROZEN_VARIANT_AUXILIARY_HTTP = 1


@dataclass(slots=True)
class VariantDispatchLedger:
    """Dispatch counters for the instruction-variant slice.

    Attributes:
        judgment_calls (int): Completed ``judge`` invocations.
        scoring_requests (int): Completed ``score_candidates`` calls.
        failed_attempts (int): Attempts that raised before a durable row.

    Examples:
        ```python
        ledger = VariantDispatchLedger()
        ledger.before_judgment()
        assert ledger.judgment_calls == 1
        ```
    """

    judgment_calls: int = 0
    scoring_requests: int = 0
    failed_attempts: int = 0

    def before_judgment(self) -> None:
        """Reserve one judgment slot.

        Raises:
            ValueError: When the judgment budget is exhausted.
        """
        next_count = self.judgment_calls + 1
        if next_count > FROZEN_VARIANT_JUDGMENT_CALLS:
            msg = (
                f"judgment calls {next_count} exceed budget "
                f"{FROZEN_VARIANT_JUDGMENT_CALLS}"
            )
            raise ValueError(msg)
        self.judgment_calls = next_count

    def before_scoring(self) -> None:
        """Reserve one scoring slot.

        Raises:
            ValueError: When the scoring budget is exhausted.
        """
        next_count = self.scoring_requests + 1
        if next_count > FROZEN_VARIANT_SCORING_REQUESTS:
            msg = (
                f"scoring requests {next_count} exceed budget "
                f"{FROZEN_VARIANT_SCORING_REQUESTS}"
            )
            raise ValueError(msg)
        self.scoring_requests = next_count

    def record_failure(self) -> None:
        """Increment failed attempts (invalid inputs, early abort)."""
        self.failed_attempts += 1


def plan_instruction_variant_calls() -> ConsumerCallCounts:
    """Return scheduled judgment and scoring totals for two variants on two cases.

    Returns:
        Frozen plan: four ``judge`` calls and four Noul scoring requests
        (hard cap eight scoring requests across both variants).
    """
    return ConsumerCallCounts(
        judgment_calls=FROZEN_VARIANT_JUDGMENT_CALLS,
        scoring_requests=4,
        auxiliary_http=FROZEN_VARIANT_AUXILIARY_HTTP,
    )
