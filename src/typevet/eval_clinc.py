"""CLINC150 ``plus`` loader with domain-sharded 15-way Choice.

clinc/clinc_oos (CC BY 3.0) assigns one global intent slug per utterance.
typevet loads **one domain at a time** (15 Choice labels from ``domains.json``).
Optional OOS rows use **Noul** ``in_scope``; separate ontology from Banking77 proxy.

Examples:
    Load the banking shard from vendored micro JSONL (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.eval_clinc import DEFAULT_DOMAIN, load_plus_split

    jsonl = Path("tests/fixtures/clinc/plus_banking_micro.jsonl").read_text()
    rows = load_plus_split(domain=DEFAULT_DOMAIN, jsonl_text=jsonl)
    assert len(rows) == 15
    ```

See Also:
    - docs/reference/eval-clinc-shard-map.md
    - [typevet.eval_clinc_download][]: Hugging Face datasets-server stream
    - [typevet.eval_clinc_shard][]: domain catalog and schema fixtures
    - [typevet.eval_clinc_rows][]: row mapping and balancing helpers
    - docs/reference/banking77-proxy-and-metrics.md (separate corpus; no shared enum)
"""

from __future__ import annotations

import httpx

from typevet.eval_clinc_download import download_plus_jsonl
from typevet.eval_clinc_rows import (
    ClincExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    intent_slug_from_record,
    iter_plus_rows,
    map_examples,
    map_row,
)
from typevet.eval_clinc_shard import (
    BANNED_STATE_KEYS,
    CONFIG,
    DATASET_ID,
    DEFAULT_DOMAIN,
    IN_SCOPE_NOUL_SCHEMA,
    NOUL_LABELS,
    OOS_INTENT,
    PRIMARY_CHOICE_NAME,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    choice_labels_for_domain,
    choice_schema_for_domain,
    domain_intent_map,
    domain_keys,
    plus_intent_names,
)

__all__ = [
    "BANNED_STATE_KEYS",
    "CONFIG",
    "DATASET_ID",
    "DEFAULT_DOMAIN",
    "IN_SCOPE_NOUL_SCHEMA",
    "NOUL_LABELS",
    "OOS_INTENT",
    "PRIMARY_CHOICE_NAME",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "ClincExample",
    "assert_state_keys_allowed",
    "balanced_sample",
    "build_state",
    "choice_labels_for_domain",
    "choice_schema_for_domain",
    "domain_intent_map",
    "domain_keys",
    "intent_slug_from_record",
    "load_plus_split",
    "map_row",
    "plus_intent_names",
]


def load_plus_split(
    *,
    domain: str = DEFAULT_DOMAIN,
    include_oos: bool = False,
    limit: int | None = None,
    seed: int = 0,
    balanced: bool = False,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[ClincExample]:
    """Load CLINC ``plus`` train rows for one domain shard.

    Args:
        domain: CLINC domain key (default ``banking``).
        include_oos: Retain global ``oos`` utterances with Noul gold.
        limit: Optional row cap (applied before balancing when ``balanced``).
        seed: Seed for ``balanced_sample`` ordering.
        balanced: When true, equalize per-intent counts before applying ``limit``.
        jsonl_text: Offline JSONL body; when omitted, download from Hugging Face.
        client: Optional ``httpx`` client for Hub download.

    Returns:
        Mapped [ClincExample][] rows for the selected domain.

    Raises:
        ValueError: Unknown domain or invalid shard geometry.
    """
    choice_labels_for_domain(domain)
    text = jsonl_text if jsonl_text is not None else download_plus_jsonl(client=client)
    examples = map_examples(
        iter_plus_rows(text), domain=domain, include_oos=include_oos
    )
    if balanced:
        return balanced_sample(examples, domain, limit, seed)
    if limit is not None:
        return examples[:limit]
    return examples
