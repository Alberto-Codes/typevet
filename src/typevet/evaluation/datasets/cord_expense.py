"""CORD expense smoke: amount normalization, claim cases and routing ([#164][i164]).

The vendored manifest under ``tests/fixtures/cord/expense_smoke/`` pins six
CORD v2 validation receipts ([#163][i163]). Each receipt holds three synthetic
claims: one states the annotated total, one states a different total and one
masks digits so no total can be read.

A judge answers one three-label ``Choice``. The labels keep a pinned order,
``insufficient_evidence``, ``mismatch``, ``match``, and ``route`` maps each
label to exactly one manifest verdict. Gold stays on the case and never enters
``ExpenseCase.model_inputs``.

The question wording makes a legible claimed amount the precondition for
``mismatch`` and ``match`` ([#182][i182]). A masked claim writes ``?`` in place
of a digit while the receipt total stays readable, so the earlier wording let a
literal reader compare anyway and ``insufficient_evidence`` never won.

[i182]: https://github.com/Alberto-Codes/typevet/issues/182

[i163]: https://github.com/Alberto-Codes/typevet/issues/163
[i164]: https://github.com/Alberto-Codes/typevet/issues/164

Examples:
    Load the cases and route a judge label:

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.cord_expense import (
        MATCH,
        load_expense_cases,
        normalize_amount,
        route,
    )

    root = Path("tests/fixtures/cord/expense_smoke")
    cases = load_expense_cases((root / "manifest.json").read_text())
    assert len(cases) == 18
    assert normalize_amount("80,500") == "80500"
    assert route(MATCH) == "supported"
    ```

See Also:
    - [typevet.evaluation.datasets.psai_vision][]: sister vendored image smoke
    - [typevet.domain.judgment_questions][]: the ``Choice`` question type

Attributes:
    MANIFEST_VERSION (int): Manifest shape this module reads.
    DATASET_LICENSE (str): Licence every vendored receipt carries.
    INSUFFICIENT_EVIDENCE (str): Label for a claim the receipt cannot settle.
    MISMATCH (str): Label for a claim that disagrees with the receipt total.
    MATCH (str): Label for a claim that agrees with the receipt total.
    LABEL_ORDER (tuple[str, ...]): The three labels in pinned order.
    SUPPORTED (str): Manifest verdict for an agreeing claim.
    CONTRADICTED (str): Manifest verdict for a disagreeing claim.
    INSUFFICIENT (str): Manifest verdict for an unreadable claim.
    VERDICTS (tuple[str, ...]): Every manifest verdict.
    CLAIM_PROVENANCE (str): Provenance tag for a claim that states an amount.
    DAMAGE_PROVENANCE (str): Provenance tag for a claim with masked digits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet.domain.judgment_questions import Choice

MANIFEST_VERSION: Final[int] = 1
DATASET_LICENSE: Final[str] = "CC-BY-4.0"

INSUFFICIENT_EVIDENCE: Final[str] = "insufficient_evidence"
MISMATCH: Final[str] = "mismatch"
MATCH: Final[str] = "match"
LABEL_ORDER: Final[tuple[str, ...]] = (INSUFFICIENT_EVIDENCE, MISMATCH, MATCH)

SUPPORTED: Final[str] = "supported"
CONTRADICTED: Final[str] = "contradicted"
INSUFFICIENT: Final[str] = "insufficient"
VERDICTS: Final[tuple[str, ...]] = (SUPPORTED, CONTRADICTED, INSUFFICIENT)

CLAIM_PROVENANCE: Final[str] = "synthetic_claim"
DAMAGE_PROVENANCE: Final[str] = "synthetic_damage"

_ROUTES: Final[Mapping[str, str]] = {
    INSUFFICIENT_EVIDENCE: INSUFFICIENT,
    MISMATCH: CONTRADICTED,
    MATCH: SUPPORTED,
}
_CRITERIA: Final[Mapping[str, str]] = {
    INSUFFICIENT_EVIDENCE: (
        "The claimed amount is not fully readable (for example it holds a ? "
        "in place of a digit), or the receipt total cannot be read."
    ),
    MISMATCH: (
        "The claim shows every digit of its amount "
        "and that amount differs from the receipt total."
    ),
    MATCH: (
        "The claim shows every digit of its amount "
        "and that amount equals the receipt total."
    ),
}
_INSTRUCTIONS: Final[str] = (
    "Look at the claimed amount in the expense claim before you look at the "
    "receipt. A ? stands where a digit was lost; it is not a digit you may "
    "guess and it is not a wildcard. Compare with the receipt total only when "
    "the claim shows every digit of its amount. Otherwise choose "
    "insufficient_evidence. A comma or a dot groups digits and does not "
    "change an amount."
)
_PLAIN = re.compile(r"(?:0|[1-9]\d*)")
_GROUPED = re.compile(r"[1-9]\d{0,2}(?P<sep>[,.])\d{3}(?:(?P=sep)\d{3})*")

_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "manifest_version",
    "dataset_id",
    "dataset_revision",
    "license",
    "split",
    "source_file",
    "source_file_sha256",
    "receipts",
)
_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "receipt_id",
    "row_idx",
    "image_id",
    "image",
    "annotated_total",
    "normalized_total",
    "claims",
)
_IMAGE_KEYS: Final[tuple[str, ...]] = ("file_name", "mime_type", "sha256")
_CLAIM_KEYS: Final[tuple[str, ...]] = (
    "claim_id",
    "expected_verdict",
    "statement",
    "claimed_amount",
    "provenance",
)


def normalize_amount(text: str) -> str:
    """Return a whole-unit amount with its thousands separators removed.

    CORD receipts are IDR-style: whole units, with a comma or a dot between
    groups of three digits. One amount uses one separator throughout.

    Args:
        text: Amount as printed, for example ``80,500`` or ``72.000``.

    Returns:
        Digits only, for example ``80500``.

    Raises:
        ValueError: When the text is masked, signed, has a currency prefix,
            mixes separators or has a group that is not three digits.
    """
    value = text.strip()
    if _PLAIN.fullmatch(value):
        return value
    if _GROUPED.fullmatch(value):
        return re.sub(r"[,.]", "", value)
    msg = f"CORD amount {text!r} is not a whole amount with thousands separators"
    raise ValueError(msg)


def expense_question() -> Choice:
    """Build the three-label expense question with labels in pinned order.

    Returns:
        A new ``Choice`` whose criteria keys follow ``LABEL_ORDER``.
    """
    return Choice(
        criteria={label: _CRITERIA[label] for label in LABEL_ORDER},
        instructions=_INSTRUCTIONS,
    )


def route(label: str) -> str:
    """Map one judge label to its manifest verdict.

    Args:
        label: One of ``LABEL_ORDER``.

    Returns:
        ``insufficient``, ``contradicted`` or ``supported``.

    Raises:
        ValueError: When ``label`` is not a label of the expense question.
    """
    try:
        return _ROUTES[label]
    except KeyError:
        allowed = ", ".join(LABEL_ORDER)
        msg = f"CORD expense label {label!r} not in {allowed}"
        raise ValueError(msg) from None


@dataclass(frozen=True, slots=True)
class ExpenseCase:
    """One synthetic claim against one vendored receipt.

    Attributes:
        claim_id (str): Claim key, ``<receipt_id>-C<n>``.
        receipt_id (str): Receipt key, ``R01`` to ``R06``.
        statement (str): Claim text the judge reads.
        expected_verdict (str): Gold verdict. Never part of a model input.
        image_file_name (str): Receipt PNG beside the manifest.
        image_mime_type (str): Always ``image/png``.
        image_sha256 (str): Digest of the vendored PNG bytes.
        source (Mapping[str, Any]): Corpus provenance for this case.

    Examples:
        ```python
        from pathlib import Path

        from typevet.evaluation.datasets.cord_expense import load_expense_cases

        root = Path("tests/fixtures/cord/expense_smoke")
        case = load_expense_cases((root / "manifest.json").read_text())[0]
        assert case.model_inputs() == {"statement": case.statement}
        ```
    """

    claim_id: str
    receipt_id: str
    statement: str
    expected_verdict: str
    image_file_name: str
    image_mime_type: str
    image_sha256: str
    source: Mapping[str, Any]

    def model_inputs(self) -> dict[str, str]:
        """Return the text a judge may see. Gold is never included.

        Returns:
            Mapping with the claim ``statement`` only.
        """
        return {"statement": self.statement}

    def provenance(self) -> dict[str, Any]:
        """Return corpus and claim provenance for this case.

        Returns:
            Copy of the dataset, revision, file, row, image and claim tags.
        """
        return dict(self.source)


def _as_object(raw: Any, where: str) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    msg = f"CORD expense {where} must be an object"
    raise ValueError(msg)


def _as_array(raw: Any, where: str) -> Sequence[Any]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return raw
    msg = f"CORD expense {where} must be an array"
    raise ValueError(msg)


def _require_keys(raw: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        msg = f"CORD expense {where} missing required keys: {', '.join(missing)}"
        raise ValueError(msg)


def _reference_verdict(claimed: Any, total: str) -> str:
    if claimed is None:
        return INSUFFICIENT
    return SUPPORTED if normalize_amount(str(claimed)) == total else CONTRADICTED


def _check_claim(raw: Mapping[str, Any], total: str) -> None:
    cid = str(raw["claim_id"])
    verdict = str(raw["expected_verdict"])
    reference = _reference_verdict(raw["claimed_amount"], total)
    if verdict != reference:
        msg = f"CORD expense claim {cid!r} verdict {verdict!r} should be {reference!r}"
        raise ValueError(msg)
    tag = str(raw["provenance"])
    wanted = DAMAGE_PROVENANCE if raw["claimed_amount"] is None else CLAIM_PROVENANCE
    if tag != wanted:
        msg = f"CORD expense claim {cid!r} provenance {tag!r} should be {wanted!r}"
        raise ValueError(msg)


def _receipt_cases(raw: Any, dataset: Mapping[str, Any]) -> tuple[ExpenseCase, ...]:
    receipt = _as_object(raw, "receipt")
    _require_keys(receipt, _RECEIPT_KEYS, "receipt")
    rid = str(receipt["receipt_id"])
    image = _as_object(receipt["image"], f"receipt {rid!r} image")
    _require_keys(image, _IMAGE_KEYS, f"receipt {rid!r} image")
    total = str(receipt["normalized_total"])
    if normalize_amount(str(receipt["annotated_total"])) != total:
        msg = f"CORD expense receipt {rid!r} normalized_total {total!r} is wrong"
        raise ValueError(msg)
    cases: list[ExpenseCase] = []
    for entry in _as_array(receipt["claims"], f"receipt {rid!r} claims"):
        claim = _as_object(entry, f"receipt {rid!r} claim")
        _require_keys(claim, _CLAIM_KEYS, f"receipt {rid!r} claim")
        _check_claim(claim, total)
        source = {
            **dataset,
            "receipt_id": rid,
            "row_idx": int(receipt["row_idx"]),
            "image_id": int(receipt["image_id"]),
            "image_sha256": str(image["sha256"]),
            "claim_provenance": str(claim["provenance"]),
        }
        cases.append(
            ExpenseCase(
                claim_id=str(claim["claim_id"]),
                receipt_id=rid,
                statement=str(claim["statement"]),
                expected_verdict=str(claim["expected_verdict"]),
                image_file_name=str(image["file_name"]),
                image_mime_type=str(image["mime_type"]),
                image_sha256=str(image["sha256"]),
                source=source,
            )
        )
    return tuple(cases)


def _dataset_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(raw, _MANIFEST_KEYS, "manifest")
    version = raw["manifest_version"]
    if version != MANIFEST_VERSION:
        msg = f"CORD expense manifest_version {version!r} is not {MANIFEST_VERSION}"
        raise ValueError(msg)
    if raw["license"] != DATASET_LICENSE:
        msg = f"CORD expense license {raw['license']!r} must be {DATASET_LICENSE!r}"
        raise ValueError(msg)
    keys = ("dataset_id", "dataset_revision", "license", "split", "source_file")
    fields: dict[str, Any] = {key: str(raw[key]) for key in keys}
    fields["source_file_sha256"] = str(raw["source_file_sha256"])
    return fields


def load_expense_cases(manifest_text: str) -> tuple[ExpenseCase, ...]:
    """Parse the vendored manifest into one case per claim, in manifest order.

    Every claim verdict is checked against its claimed amount and the
    normalized receipt total, so a mislabelled gold row cannot load.

    Args:
        manifest_text: UTF-8 JSON text of ``manifest.json``.

    Returns:
        Cases in receipt order, then claim order.

    Raises:
        ValueError: When the JSON, manifest shape, licence, version, totals,
            verdicts, provenance tags or claim ids are wrong.
    """
    try:
        raw = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        msg = "CORD expense manifest is not valid JSON"
        raise ValueError(msg) from exc
    manifest = _as_object(raw, "manifest")
    dataset = _dataset_fields(manifest)
    receipts = _as_array(manifest["receipts"], "receipts")
    cases = tuple(c for r in receipts for c in _receipt_cases(r, dataset))
    seen: set[str] = set()
    for case in cases:
        if case.claim_id in seen:
            msg = f"CORD expense manifest has duplicate claim_id {case.claim_id!r}"
            raise ValueError(msg)
        seen.add(case.claim_id)
    return cases
