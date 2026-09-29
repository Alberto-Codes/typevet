"""CLINC150 domain shards, intent catalog, and versioned JSON Schema fixtures.

Vendored ``clinc_domains.json`` (upstream ``domains.json``) defines ten
15-intent shards. ``clinc_plus_intent_names.json`` lists global ``plus`` intent
slugs in Hub index order.

Examples:
    Resolve banking Choice labels and schema from vendored shard JSON:

    ```python
    from typevet_evals.datasets.clinc_shard import (
        DEFAULT_DOMAIN,
        choice_labels_for_domain,
        choice_schema_for_domain,
    )

    labels = choice_labels_for_domain(DEFAULT_DOMAIN)
    assert len(labels) == 15
    schema = choice_schema_for_domain(DEFAULT_DOMAIN)
    assert "intent" in schema["properties"]
    ```

See Also:
    - [typevet_evals.datasets.clinc][]: public loader facade
    - docs/reference/eval-clinc-shard-map.md
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

SOURCE: Final[str] = "clinc/clinc_oos"
DATASET_ID: Final[str] = SOURCE
CONFIG: Final[str] = "plus"
SPLIT: Final[str] = "train"
DEFAULT_DOMAIN: Final[str] = "banking"
OOS_INTENT: Final[str] = "oos"
INTENTS_PER_DOMAIN: Final[int] = 15
PRIMARY_CHOICE_NAME: Final[str] = "intent"
PRIMARY_NOUL_NAME: Final[str] = "in_scope"
NOUL_LABELS: Final[tuple[str, ...]] = ("yes", "no")
BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {"intent", "expected", "label", "ground_truth", "answer_key", "domain"}
)

_PACKAGE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def domain_intent_map() -> dict[str, tuple[str, ...]]:
    """Return domain name to ordered in-domain intent slug tuples.

    Returns:
        Mapping from CLINC domain keys to 15-intent slug tuples.
    """
    raw = json.loads(
        _PACKAGE_DIR.joinpath("clinc_domains.json").read_text(encoding="utf-8")
    )
    return {domain: tuple(slugs) for domain, slugs in raw.items()}


@lru_cache(maxsize=1)
def plus_intent_names() -> tuple[str, ...]:
    """Return global ``plus`` intent slugs in Hugging Face index order.

    Returns:
        Tuple of 150 intent slug strings plus the global ``oos`` slug.
    """
    raw = json.loads(
        _PACKAGE_DIR.joinpath("clinc_plus_intent_names.json").read_text(
            encoding="utf-8"
        )
    )
    return tuple(str(name) for name in raw)


def domain_keys() -> tuple[str, ...]:
    """Return sorted CLINC domain keys present in the shard map.

    Returns:
        Sorted domain names (ten shards in v1).
    """
    return tuple(sorted(domain_intent_map()))


def choice_labels_for_domain(domain: str) -> tuple[str, ...]:
    """Return the 15 Choice labels for one domain shard.

    Args:
        domain: CLINC domain key (for example ``banking``).

    Returns:
        Intent slugs allowed for in-domain Choice gold in that shard.

    Raises:
        ValueError: Unknown domain or shard size other than 15.
    """
    try:
        labels = domain_intent_map()[domain]
    except KeyError as exc:
        msg = f"Unknown CLINC domain {domain!r}; expected one of {', '.join(domain_keys())}"
        raise ValueError(msg) from exc
    if len(labels) != INTENTS_PER_DOMAIN:
        msg = f"CLINC domain {domain!r} must have {INTENTS_PER_DOMAIN} intents, got {len(labels)}"
        raise ValueError(msg)
    return labels


def choice_schema_for_domain(domain: str) -> dict[str, Any]:
    """Build the primary 15-way Choice JSON Schema for a domain shard.

    Args:
        domain: CLINC domain key.

    Returns:
        Object schema with a single ``intent`` string enum property.

    Raises:
        ValueError: Passed through from ``choice_labels_for_domain``.
    """
    labels = choice_labels_for_domain(domain)
    return {
        "type": "object",
        "properties": {
            PRIMARY_CHOICE_NAME: {
                "type": "string",
                "enum": list(labels),
                "instructions": (
                    f"Given the user utterance, which in-domain intent applies "
                    f"within the {domain} shard?"
                ),
            }
        },
        "required": [PRIMARY_CHOICE_NAME],
        "additionalProperties": False,
    }


IN_SCOPE_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "string",
            "enum": list(NOUL_LABELS),
            "instructions": (
                "Is the utterance in scope for the domain shard (yes) or out of scope (no)?"
            ),
            "return_probabilities": True,
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}
