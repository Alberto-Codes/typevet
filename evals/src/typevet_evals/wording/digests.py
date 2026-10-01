"""Pin the seed and evolved wording by digest in a receipt (#362).

A reader matches a receipt to a candidate by digest, not by full text. A
candidate is a mapping of component name to text. ``component_digests``
gives one digest per component, one digest of the whole mapping and the
gepa-adk ``Candidate.id`` of the same mapping. ``wording_digests`` builds the
``wording_digests`` receipt block for the seed and the evolved mapping.
``wording_fields`` gives the verbatim texts and that block as receipt keys.

The rules:

- A component digest is the SHA-256 of the UTF-8 text.
- The mapping digest is the SHA-256 of the canonical JSON of the mapping:
  sorted keys, no spaces (``separators=(",", ":")``), ASCII escapes.
- ``gepa_candidate_id`` uses the gepa-adk rule: the first 12 hex characters
  of the SHA-256 of ``json.dumps(components, sort_keys=True,
  ensure_ascii=False)``. That JSON has spaces after the separators, so it
  is not the mapping digest.

Attributes:
    WORDING_COMPONENT (str): The component name of the wording text.

Examples:
    ```python
    from typevet_evals.wording.digests import wording_digests

    block = wording_digests("Is this a scam?", "Does it ask for money?")
    block["seed"]["components"]["instructions"]  # 64 hex characters
    ```

See Also:
    - [typevet_evals.wording.held_out][]: the held-out receipt
    - [typevet_evals.wording.comparison][]: the comparison receipt
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "WORDING_COMPONENT",
    "component_digests",
    "gepa_candidate_id",
    "mapping_digest",
    "text_digest",
    "wording_digests",
    "wording_fields",
]

WORDING_COMPONENT: Final[str] = "instructions"
_GEPA_ID_LENGTH: Final[int] = 12


def text_digest(text: str) -> str:
    """Return the SHA-256 hex digest of the UTF-8 text.

    Args:
        text: One component text.

    Returns:
        64 lowercase hex characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mapping_digest(mapping: Mapping[str, str]) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of the mapping.

    Args:
        mapping: Component name to text.

    Returns:
        64 lowercase hex characters; the key order does not change it.
    """
    canonical = json.dumps(dict(mapping), sort_keys=True, separators=(",", ":"))
    return text_digest(canonical)


def gepa_candidate_id(mapping: Mapping[str, str]) -> str:
    """Return the gepa-adk ``Candidate.id`` of a candidate with these components.

    Args:
        mapping: Component name to text.

    Returns:
        12 lowercase hex characters.
    """
    loose = json.dumps(dict(mapping), sort_keys=True, ensure_ascii=False)
    return text_digest(loose)[:_GEPA_ID_LENGTH]


def component_digests(mapping: Mapping[str, str]) -> dict[str, Any]:
    """Return the digests of one candidate mapping.

    Args:
        mapping: Component name to text.

    Returns:
        ``components`` (name to digest), ``mapping`` and ``gepa_candidate_id``.
    """
    return {
        "components": {name: text_digest(text) for name, text in mapping.items()},
        "mapping": mapping_digest(mapping),
        "gepa_candidate_id": gepa_candidate_id(mapping),
    }


def wording_digests(seed_text: str, evolved_text: str) -> dict[str, Any]:
    """Return the ``wording_digests`` receipt block.

    Each arm is the mapping ``{WORDING_COMPONENT: text}``.

    Args:
        seed_text: The seed wording.
        evolved_text: The evolved wording.

    Returns:
        ``seed`` and ``evolved``, each a ``component_digests`` block.
    """
    return {
        "seed": component_digests({WORDING_COMPONENT: seed_text}),
        "evolved": component_digests({WORDING_COMPONENT: evolved_text}),
    }


def wording_fields(seed_text: str, evolved_text: str) -> dict[str, Any]:
    """Return the wording keys of a held-out or comparison receipt.

    Args:
        seed_text: The seed wording.
        evolved_text: The evolved wording.

    Returns:
        ``seed_text`` and ``evolved_text`` verbatim, and ``wording_digests``.
    """
    return {
        "seed_text": seed_text,
        "evolved_text": evolved_text,
        "wording_digests": wording_digests(seed_text, evolved_text),
    }
