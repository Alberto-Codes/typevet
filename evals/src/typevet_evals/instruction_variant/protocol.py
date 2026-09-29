"""Revision 2 pins and attempt accounting for the instruction-variant consumer slice ([#177][i177]).

Examples:
    ```python
    from typevet_evals.instruction_variant.protocol import (
        FROZEN_VARIANT_SCORING_REQUESTS,
        plan_instruction_variant_calls,
    )

    plan = plan_instruction_variant_calls()
    assert plan.scoring_requests <= FROZEN_VARIANT_SCORING_REQUESTS
    ```

See Also:
    - [typevet_evals.instruction_variant.proof][]: harness runner
    - [typevet_evals.psai_vision_consumer.http_accounting][]: dispatch accounting
    - [docs/maintainers/consumer-instruction-variant-protocol.md][]: maintainer table

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from typing import ClassVar

from typevet_evals.psai_vision_consumer.accounting import ConsumerCallCounts
from typevet_evals.psai_vision_consumer.http_accounting import DispatchAccounting

INSTRUCTION_VARIANT_PROTOCOL_REVISION = 2

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
FROZEN_VARIANT_AUXILIARY_HTTP = 10


class VariantDispatchLedger(DispatchAccounting):
    """Variant attempts with independent success counters and frozen ceilings.

    Attributes:
        limits (tuple[int, ...]): Judgment, scoring, metadata, tokenizer, completion.

    Examples:
        ```python
        ledger = VariantDispatchLedger()
        ledger.before_judgment()
        assert ledger.judgment_attempts == 1
        ```
    """

    limits: ClassVar[tuple[int, ...]] = (4, 8, 2, 8, 8)

    def before_judgment(self) -> None:
        """Reserve an admitted judgment attempt."""
        self.before_judgment_dispatch()

    def before_scoring(self) -> None:
        """Reserve an admitted scoring attempt."""
        self.before_scoring_dispatch()

    def record_failure(self) -> None:
        """Retain a failed dispatch or offline negative probe."""
        self.record_failed_attempt()


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
