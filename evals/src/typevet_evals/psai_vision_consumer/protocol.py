"""Frozen consumer live protocol pins shared by receipts and dispatch ([#177][i177]).

Examples:
    ```python
    from typevet_evals.psai_vision_consumer.protocol import (
        expected_matrix_row_keys,
        committed_consumer_fixture_root,
    )

    assert len(expected_matrix_row_keys()) == 14
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.accounting][]: call budgets
    - [docs/maintainers/consumer-live-protocol-rev2.md][]: maintainer table

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from pathlib import Path

from typevet_evals.psai_vision_consumer.accounting import (
    FROZEN_CONSUMER_CASE_UIDS,
    TEXT_ANNOTATION_JUDGE_UIDS,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
COMMITTED_CONSUMER_FIXTURE_ROOT = _REPO_ROOT / "tests/fixtures/psai/vision_smoke"

FROZEN_PROTOCOL_REVISION = 2
FROZEN_JUDGMENT_CALLS = 14
FROZEN_SCORING_REQUESTS = 16
FROZEN_AUXILIARY_METADATA_HTTP = 3
FROZEN_AUXILIARY_TOKENIZER_HTTP_CEILING = 128


def committed_consumer_fixture_root() -> Path:
    """Return the committed vision smoke directory for replay checks.

    Returns:
        Path to ``tests/fixtures/psai/vision_smoke`` in this repository.
    """
    return COMMITTED_CONSUMER_FIXTURE_ROOT


def expected_matrix_row_keys() -> frozenset[tuple[str, str, str]]:
    """Return the frozen ``(uid, condition, leg)`` keys for protocol rev 2.

    Returns:
        Four cases times three visual conditions plus two annotation rows.
    """
    keys: set[tuple[str, str, str]] = set()
    for uid in FROZEN_CONSUMER_CASE_UIDS:
        for condition in ("present", "omitted", "swapped"):
            keys.add((uid, condition, "visual"))
    for uid in TEXT_ANNOTATION_JUDGE_UIDS:
        keys.add((uid, "text_annotation", "annotation"))
    return frozenset(keys)


def expected_paired_ordering_uids() -> frozenset[str]:
    """Return case ids that must appear in ``paired_ordering``.

    Returns:
        ``FROZEN_CONSUMER_CASE_UIDS`` as a frozenset.
    """
    return frozenset(FROZEN_CONSUMER_CASE_UIDS)
