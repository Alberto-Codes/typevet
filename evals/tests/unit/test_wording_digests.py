"""Unit checks for the wording digests block (#362).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_digests.py
    ```

See Also:
    - [typevet_evals.wording.digests][]: the digests module
"""

from __future__ import annotations

import hashlib
import json

import pytest
from gepa_adk import Candidate

from typevet_evals.wording.digests import (
    WORDING_COMPONENT,
    component_digests,
    gepa_candidate_id,
    mapping_digest,
    text_digest,
    wording_digests,
    wording_fields,
)
from typevet_evals.wording.parts import WordingParts

pytestmark = pytest.mark.unit

MAPPING = {"instructions": "Is this message a scam?", "criteria": "Café rules"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_text_digest_is_the_sha256_of_the_utf8_text() -> None:
    assert text_digest("Café") == _sha("Café")
    assert text_digest("a") != text_digest("b")


def test_mapping_digest_is_the_sha256_of_the_canonical_json() -> None:
    canonical = json.dumps(MAPPING, sort_keys=True, separators=(",", ":"))

    assert mapping_digest(MAPPING) == _sha(canonical)
    assert mapping_digest(dict(reversed(MAPPING.items()))) == mapping_digest(MAPPING)


def test_component_digests_give_one_per_part_and_one_for_the_mapping() -> None:
    block = component_digests(MAPPING)

    assert block == {
        "components": {name: _sha(text) for name, text in MAPPING.items()},
        "mapping": _sha(json.dumps(MAPPING, sort_keys=True, separators=(",", ":"))),
        "gepa_candidate_id": Candidate(components=dict(MAPPING)).id,
    }


def test_the_mapping_digest_is_not_the_gepa_candidate_id() -> None:
    candidate = Candidate(components=dict(MAPPING))

    assert gepa_candidate_id(MAPPING) == candidate.id
    assert mapping_digest(MAPPING) != candidate.id
    assert not mapping_digest(MAPPING).startswith(candidate.id)


def test_wording_digests_digest_the_seed_and_the_evolved_mapping() -> None:
    block = wording_digests(WordingParts.from_texts("seed", "evolved"))

    assert WORDING_COMPONENT == "instructions"
    assert block == {
        "seed": component_digests({"instructions": "seed"}),
        "evolved": component_digests({"instructions": "evolved"}),
    }
    json.dumps(block)


SEED_PARTS = {
    "instructions": "Is this message a scam?",
    "criteria_true": "It is a scam",
    "criteria_false": "It is légitimate",
}
EVOLVED_PARTS = SEED_PARTS | {"criteria_true": "It asks for money"}


def test_the_candidate_id_is_gepa_adks_over_the_evolved_selection() -> None:
    full = WordingParts(tuple(SEED_PARTS), SEED_PARTS, EVOLVED_PARTS)
    part = WordingParts(("criteria_true",), SEED_PARTS, EVOLVED_PARTS)

    block, selected = wording_digests(full), wording_digests(part)

    assert block["evolved"]["gepa_candidate_id"] == (
        Candidate(components=dict(EVOLVED_PARTS)).id
    )
    assert block["seed"]["gepa_candidate_id"] == Candidate(components=SEED_PARTS).id
    assert selected["evolved"]["gepa_candidate_id"] == (
        Candidate(components={"criteria_true": "It asks for money"}).id
    )
    assert selected["evolved"]["mapping"] == block["evolved"]["mapping"]
    assert selected["evolved"]["components"] == {
        name: _sha(text) for name, text in EVOLVED_PARTS.items()
    }


def test_wording_fields_carry_the_selection_and_both_full_mappings() -> None:
    parts = WordingParts(("criteria_true",), SEED_PARTS, EVOLVED_PARTS)

    fields = wording_fields(parts)

    assert fields == {
        "seed_text": SEED_PARTS["instructions"],
        "evolved_text": EVOLVED_PARTS["instructions"],
        "components": ["criteria_true"],
        "seed_parts": SEED_PARTS,
        "evolved_parts": EVOLVED_PARTS,
        "wording_digests": wording_digests(parts),
    }
    json.dumps(fields)
