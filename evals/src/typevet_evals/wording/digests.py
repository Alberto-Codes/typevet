"""Pin the seed and evolved wording by digest in a receipt (#362, #363).

A reader matches a receipt to a candidate by digest, not by full text. A
candidate is a mapping of part name to text (#363). ``component_digests``
gives one digest per part, one digest of the whole mapping and the gepa-adk
``Candidate.id`` of the evolved selection. ``wording_digests`` builds the
``wording_digests`` receipt block for the seed and the evolved mapping of a
``WordingParts``. ``wording_fields`` gives the verbatim ``instructions``
texts, the selection, both full mappings and that block as receipt keys.

The rules:

- A component digest is the SHA-256 of the UTF-8 text.
- The mapping digest is the SHA-256 of the canonical JSON of the full
  mapping: sorted keys, no spaces (``separators=(",", ":")``), ASCII escapes.
- ``gepa_candidate_id`` uses the gepa-adk rule over the evolved selection
  only, because a gepa-adk candidate holds the selected parts only: the
  first 12 hex characters of the SHA-256 of ``json.dumps(selected,
  sort_keys=True, ensure_ascii=False)``. That JSON has spaces after the
  separators, so it is not the mapping digest.

Attributes:
    WORDING_COMPONENT (str): The component name of the wording text.

Examples:
    ```python
    from typevet_evals.wording.digests import wording_digests
    from typevet_evals.wording.parts import WordingParts

    parts = WordingParts.from_texts("Is this a scam?", "Does it ask for money?")
    block = wording_digests(parts)
    block["seed"]["components"]["instructions"]  # 64 hex characters
    ```

See Also:
    - [typevet_evals.wording.held_out][]: the held-out receipt
    - [typevet_evals.wording.comparison][]: the comparison receipt
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Final

from typevet_evals.wording.parts import INSTRUCTIONS, WordingParts

__all__ = [
    "WORDING_COMPONENT",
    "component_digests",
    "gepa_candidate_id",
    "mapping_digest",
    "text_digest",
    "wording_digests",
    "wording_fields",
]

WORDING_COMPONENT: Final[str] = INSTRUCTIONS
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


def component_digests(
    mapping: Mapping[str, str], selection: Iterable[str] | None = None
) -> dict[str, Any]:
    """Return the digests of one candidate mapping.

    Args:
        mapping: Part name to text, the full mapping.
        selection: The evolved part names; None selects every part.

    Returns:
        ``components`` (name to digest) and ``mapping`` over the full
        mapping, and ``gepa_candidate_id`` over the selected parts.
    """
    names = list(mapping) if selection is None else list(selection)
    return {
        "components": {name: text_digest(text) for name, text in mapping.items()},
        "mapping": mapping_digest(mapping),
        "gepa_candidate_id": gepa_candidate_id({n: mapping[n] for n in names}),
    }


def wording_digests(parts: WordingParts) -> dict[str, Any]:
    """Return the ``wording_digests`` receipt block.

    Args:
        parts: The selection and the seed and evolved full mappings.

    Returns:
        ``seed`` and ``evolved``, each a ``component_digests`` block over
        ``parts.components``.
    """
    return {
        "seed": component_digests(parts.seed, parts.components),
        "evolved": component_digests(parts.evolved, parts.components),
    }


def wording_fields(parts: WordingParts) -> dict[str, Any]:
    """Return the wording keys of a held-out, comparison or evolution record.

    Args:
        parts: The selection and the seed and evolved full mappings.

    Returns:
        ``seed_text`` and ``evolved_text`` (the ``instructions`` parts,
        verbatim), ``components``, ``seed_parts``, ``evolved_parts`` and
        ``wording_digests``.
    """
    return {
        "seed_text": parts.seed_text,
        "evolved_text": parts.evolved_text,
        "components": list(parts.components),
        "seed_parts": dict(parts.seed),
        "evolved_parts": dict(parts.evolved),
        "wording_digests": wording_digests(parts),
    }
