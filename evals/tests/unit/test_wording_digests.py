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
)

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
    block = wording_digests("seed", "evolved")

    assert WORDING_COMPONENT == "instructions"
    assert block == {
        "seed": component_digests({"instructions": "seed"}),
        "evolved": component_digests({"instructions": "evolved"}),
    }
    json.dumps(block)
