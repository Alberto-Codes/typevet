"""Expected-outcome checks for consumer receipts ([#177][i177]).

Examples:
    ```python
    from typevet_evals.psai_vision_consumer.outcomes import (
        expected_outcome_failures,
    )

    assert expected_outcome_failures({}) == []
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.receipt][]: acceptance wrapper

Gold replay resolves the committed fixture from receipt identity pins
(``manifest_sha256``, ``frozen_case_image_digests``) when ``fixture_root``
is missing, for example on an isolated wheel install.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from typevet.evaluation.datasets.psai_vision import (
    VisionSmokeFixture,
    load_vision_smoke,
    vision_smoke_manifest_path,
)
from typevet.evaluation.datasets.psai_vision_controls import (
    VISUAL_QUESTION_NAME,
    VisualControl,
    control_matrix,
    noul_polarity,
    semantic_hit,
)
from typevet_evals.psai_vision_consumer.accounting import FROZEN_CONSUMER_CASE_UIDS
from typevet_evals.psai_vision_consumer.protocol import (
    committed_consumer_fixture_root,
)


def _load_frozen_controls(
    fixture_root: Path,
) -> tuple[VisionSmokeFixture, tuple[VisualControl, ...]]:
    manifest_path = vision_smoke_manifest_path(fixture_root)
    if not manifest_path.is_file():
        msg = f"missing vision smoke manifest: {manifest_path}"
        raise FileNotFoundError(msg)
    fixture = load_vision_smoke(manifest_path.read_text(encoding="utf-8"))
    controls = tuple(
        control
        for control in control_matrix(fixture.examples)
        if control.unique_data_id in FROZEN_CONSUMER_CASE_UIDS
    )
    if len({control.unique_data_id for control in controls}) != len(
        FROZEN_CONSUMER_CASE_UIDS
    ):
        msg = "manifest missing one or more frozen consumer case rows"
        raise ValueError(msg)
    return fixture, controls


def _visual_row_failures(
    row: Mapping[str, Any],
    control_by_key: Mapping[tuple[str, str], VisualControl],
) -> list[str]:
    failures: list[str] = []
    uid = row.get("unique_data_id")
    condition = row.get("condition")
    if not isinstance(uid, str) or not isinstance(condition, str):
        return failures
    control = control_by_key.get((uid, condition))
    if control is None:
        return failures
    answers = row.get("answers")
    if not isinstance(answers, Mapping):
        return [f"visual row {uid}/{condition} missing answers"]
    visual = answers.get(VISUAL_QUESTION_NAME)
    if not isinstance(visual, Mapping) or visual.get("kind") != "Noul":
        failures.append(f"visual row {uid}/{condition} missing Noul answer")
        return failures
    prob = float(visual["noul"])
    if condition == "omitted" and semantic_hit(control, prob):
        failures.append(f"omitted row {uid!r} credited as semantic hit")
    elif control.counts_as_semantic_hit and not semantic_hit(control, prob):
        failures.append(f"visual row {uid}/{condition} polarity mismatch vs gold")
    return failures


def _annotation_row_failures(
    row: Mapping[str, Any],
    expected_by_uid: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    uid = row.get("unique_data_id")
    if not isinstance(uid, str):
        return []
    expected = expected_by_uid.get(uid)
    if expected is None:
        return []
    answers = row.get("answers")
    if not isinstance(answers, Mapping):
        return [f"annotation {uid} missing answers"]
    failures: list[str] = []
    category = answers.get("category")
    if isinstance(category, Mapping) and category.get("kind") == "Choice":
        if category.get("choice") != expected["category"]:
            failures.append(
                f"annotation {uid} category "
                f"{category.get('choice')!r} != {expected['category']!r}"
            )
    else:
        failures.append(f"annotation {uid} missing Choice category answer")
    login = answers.get("requires_login")
    if isinstance(login, Mapping) and login.get("kind") == "Noul":
        polarity = noul_polarity(float(login["noul"]))
        if polarity is not expected["requires_login"]:
            failures.append(f"annotation {uid} requires_login polarity != gold")
    else:
        failures.append(f"annotation {uid} missing Noul requires_login answer")
    return failures


def _manifest_at_root_matches(fixture_root: Path, manifest_pin: str) -> bool:
    manifest_path = vision_smoke_manifest_path(fixture_root)
    if not manifest_path.is_file():
        return False
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return digest == manifest_pin


def _image_pins_match_fixture(
    fixture: VisionSmokeFixture,
    image_pins: list[Any],
) -> bool:
    pin_by_uid = {
        row.get("unique_data_id"): row.get("sha256")
        for row in image_pins
        if isinstance(row, Mapping)
    }
    for example in fixture.examples:
        if example.unique_data_id not in FROZEN_CONSUMER_CASE_UIDS:
            continue
        if pin_by_uid.get(example.unique_data_id) != example.screenshot.sha256:
            return False
    return True


def _fixture_root_for_receipt(receipt: Mapping[str, Any]) -> Path | None:
    """Resolve a fixture root when manifest and image pins match on disk.

    Returns:
        Path when pins match a readable fixture tree; ``None`` when pins fail.
    """
    manifest_pin = receipt.get("manifest_sha256")
    if not isinstance(manifest_pin, str) or not manifest_pin.strip():
        return None
    image_pins = receipt.get("frozen_case_image_digests")
    if not isinstance(image_pins, list):
        return None
    candidates: list[Path] = []
    fixture_root = receipt.get("fixture_root")
    if isinstance(fixture_root, str) and fixture_root.strip():
        candidates.append(Path(fixture_root))
    committed = committed_consumer_fixture_root()
    if committed not in candidates:
        candidates.append(committed)
    for root in candidates:
        if not _manifest_at_root_matches(root, manifest_pin):
            continue
        manifest_path = vision_smoke_manifest_path(root)
        try:
            fixture = load_vision_smoke(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if _image_pins_match_fixture(fixture, image_pins):
            return root
    return None


def expected_outcome_failures(receipt: Mapping[str, Any]) -> list[str]:
    """Compare matrix rows against frozen gold expectations.

    Args:
        receipt: Consumer proof receipt dict.

    Returns:
        Human-readable failure messages (empty when outcomes match). When
        identity pins are missing or invalid, returns a single fail-closed
        message instead of skipping semantic checks.

    """
    fixture_root = _fixture_root_for_receipt(receipt)
    if fixture_root is None:
        return ["fixture identity pins missing or do not match committed fixture"]
    try:
        fixture, controls = _load_frozen_controls(fixture_root)
    except (FileNotFoundError, ValueError):
        return ["could not reload fixture for expected-outcome checks"]

    control_by_key = {(c.unique_data_id, c.condition): c for c in controls}
    expected_by_uid = {ex.unique_data_id: ex.expected() for ex in fixture.examples}
    failures: list[str] = []
    for row in receipt.get("matrix_rows") or []:
        if not isinstance(row, Mapping):
            continue
        leg = row.get("leg")
        if leg == "visual":
            failures.extend(_visual_row_failures(row, control_by_key))
        elif leg == "annotation":
            failures.extend(_annotation_row_failures(row, expected_by_uid))

    expected_scoring = int(receipt.get("scoring_request_count", 0))
    observed = int(receipt.get("scoring_requests_observed", 0))
    if expected_scoring and observed != expected_scoring:
        failures.append(
            f"scoring_requests_observed {observed} != scoring_request_count "
            f"{expected_scoring}"
        )
    return failures
