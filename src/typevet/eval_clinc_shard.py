"""Compatibility shim for the CLINC shard map (#147).

Prefer importing from ``typevet.evaluation.datasets.clinc_shard`` in new code.

Examples:
    ```python
    from typevet.eval_clinc_shard import choice_labels_for_domain
    ```

See Also:
    - [typevet.evaluation.datasets.clinc_shard][]: New home for this module
"""

from typevet.evaluation.datasets.clinc_shard import (
    BANNED_STATE_KEYS,
    CONFIG,
    DATASET_ID,
    DEFAULT_DOMAIN,
    IN_SCOPE_NOUL_SCHEMA,
    INTENTS_PER_DOMAIN,
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
    "INTENTS_PER_DOMAIN",
    "IN_SCOPE_NOUL_SCHEMA",
    "NOUL_LABELS",
    "OOS_INTENT",
    "PRIMARY_CHOICE_NAME",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "choice_labels_for_domain",
    "choice_schema_for_domain",
    "domain_intent_map",
    "domain_keys",
    "plus_intent_names",
]
