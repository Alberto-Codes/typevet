"""Manifest loader for the PSAI claim-about-screen evidence pilot v1 ([#192][i192]).

The pilot asks whether one short claim about a computer-use screenshot is
``supported``, ``contradicted`` or ``insufficient_evidence``. The design lives
in [#189 revision 4][r4]. This module loads its manifest offline. It checks
the JSON schema first, then the cross-row rules below, and raises one
``ManifestError`` that names the first rule that fails.

| Rule | Check |
|---|---|
| ``schema`` | The manifest matches ``schema.json``. |
| ``split_counts`` | Each split holds 8 baseline, 8 claim_axis, 4 swap, 4 crop. |
| ``split_isolation`` | A host or donor belongs to one split only. |
| ``fixture_overlap`` | No ``unique_data_id`` is in other PSAI fixtures. |
| ``gold_per_class`` | Each split holds at least 4 gold per class. |
| ``human_decision`` | Final, swap and crop rows carry a human decision. |
| ``verified_agreement`` | A non-human label has Gemma, construction and Qwen agreement. |
| ``input_leak`` | No host or ``task_name`` text is in a model-input field. |

Rejected items sit in their own list with reason codes. They are not rows and
no rule counts them.

[i192]: https://github.com/Alberto-Codes/typevet/issues/192
[r4]: https://github.com/Alberto-Codes/typevet/issues/189#issuecomment-5875661660

Examples:
    Load the synthetic manifest:

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.psai_evidence_pilot import load_manifest

    root = Path("tests/fixtures/psai/evidence_pilot_v1")
    manifest = load_manifest(root / "synthetic" / "manifest.json")
    assert len(manifest.rows) == 72
    ```

See Also:
    - [typevet.evaluation.datasets.psai_vision][]: earlier PSAI screenshot fixtures
    - tests/fixtures/psai/evidence_pilot_v1/schema.json: the manifest schema

Attributes:
    PILOT_DIR_NAME (str): Directory name that marks the pilot root.
    SPLITS (tuple[str, ...]): Split names in order.
    LABELS (tuple[str, ...]): Gold label classes.
    SPLIT_KIND_COUNTS (dict[str, int]): Required rows per kind in each split.
    MIN_GOLD_PER_CLASS (int): Smallest gold count per class in each split.
    QWEN_MAJORITY (int): Qwen runs that must agree with a non-human label.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

PILOT_DIR_NAME: Final = "evidence_pilot_v1"
SPLITS: Final = ("dev", "prompt_selection", "final")
LABELS: Final = ("supported", "contradicted", "insufficient_evidence")
SPLIT_KIND_COUNTS: Final = {"baseline": 8, "claim_axis": 8, "swap": 4, "crop": 4}
MIN_GOLD_PER_CLASS: Final = 4
QWEN_MAJORITY: Final = 4

_TEXT_SUFFIXES: Final = frozenset({".json", ".jsonl", ".txt", ".csv", ".md"})
_HUMAN_KINDS: Final = frozenset({"swap", "crop"})

Row = dict[str, Any]


class ManifestError(ValueError):
    """A manifest broke one named rule.

    Attributes:
        rule (str): Name of the rule that failed, for example ``split_counts``.

    Examples:
        ```python
        error = ManifestError("split_counts", "dev has 23 rows")
        assert error.rule == "split_counts"
        ```
    """

    def __init__(self, rule: str, detail: str) -> None:
        """Store the rule name and build the message ``"<rule>: <detail>"``.

        Args:
            rule: Name of the rule that failed.
            detail: What broke the rule, naming the case or split.
        """
        super().__init__(f"{rule}: {detail}")
        self.rule = rule


@dataclass(frozen=True)
class PilotManifest:
    """A loaded manifest that passed every rule.

    Attributes:
        rows (tuple[dict[str, Any], ...]): Counted rows in file order.
        rejected (tuple[dict[str, Any], ...]): Rejected items with reason codes.

    Examples:
        ```python
        final_rows = manifest.split_rows("final")
        ```
    """

    rows: tuple[Row, ...]
    rejected: tuple[Row, ...]

    def split_rows(self, split: str) -> tuple[Row, ...]:
        """Return the rows of one split.

        Args:
            split: One of ``SPLITS``.

        Returns:
            The rows whose ``split`` equals ``split``, in file order.
        """
        return tuple(row for row in self.rows if row["split"] == split)


def load_manifest(path: Path, *, pilot_root: Path | None = None) -> PilotManifest:
    """Load a pilot manifest and check the schema and every cross-row rule.

    Args:
        path: The manifest JSON file.
        pilot_root: The ``evidence_pilot_v1`` directory that holds
            ``schema.json``. Its parent is the PSAI fixture tree that the
            ``fixture_overlap`` rule scans. Defaults to the nearest ancestor
            of ``path`` with that name.

    Returns:
        The rows and rejected items.

    Raises:
        ManifestError: When the pilot root is missing (rule ``pilot_root``) or
            the manifest breaks a rule; ``rule`` names it.
    """
    root = pilot_root if pilot_root is not None else _find_pilot_root(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    error = best_match(Draft202012Validator(schema).iter_errors(data))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path)
        raise ManifestError("schema", f"at /{where}: {error.message}")
    rows: list[Row] = data["rows"]
    _check_split_counts(rows)
    _check_split_isolation(rows)
    _check_fixture_overlap(rows, root)
    _check_gold_per_class(rows)
    _check_human_decision(rows)
    _check_verified_agreement(rows)
    _check_input_leak(rows)
    return PilotManifest(rows=tuple(rows), rejected=tuple(data["rejected"]))


def _find_pilot_root(path: Path) -> Path:
    """Return the nearest ``evidence_pilot_v1`` ancestor of ``path``.

    Args:
        path: The manifest file.

    Returns:
        The pilot root directory.

    Raises:
        ManifestError: With rule ``pilot_root`` when no ancestor matches.
    """
    for parent in path.resolve().parents:
        if parent.name == PILOT_DIR_NAME:
            return parent
    raise ManifestError("pilot_root", f"no {PILOT_DIR_NAME} ancestor of {path}")


def _check_split_counts(rows: list[Row]) -> None:
    """Require the per-split row kind counts from revision 4 section 3.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``split_counts`` when a split has other counts.
    """
    for split in SPLITS:
        kinds = Counter(row["row_kind"] for row in rows if row["split"] == split)
        if dict(kinds) != SPLIT_KIND_COUNTS:
            raise ManifestError(
                "split_counts", f"{split} has {dict(kinds)}, need {SPLIT_KIND_COUNTS}"
            )


def _check_split_isolation(rows: list[Row]) -> None:
    """Keep each host in one split and take donors from the same split.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``split_isolation`` when a host or donor crosses splits.
    """
    host_splits: defaultdict[str, set[str]] = defaultdict(set)
    split_ids: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        host_splits[row["host"]].add(row["split"])
        split_ids[row["split"]].add(row["unique_data_id"])
    for row in rows:
        donor = row["donor"]
        if donor is None:
            continue
        host_splits[donor["host"]].add(row["split"])
        if donor["unique_data_id"] not in split_ids[row["split"]]:
            raise ManifestError(
                "split_isolation", f"{row['case_id']} donor is not in {row['split']}"
            )
        if donor["host"] == row["host"]:
            raise ManifestError(
                "split_isolation", f"{row['case_id']} donor shares its host"
            )
    for host, splits in sorted(host_splits.items()):
        if len(splits) > 1:
            raise ManifestError("split_isolation", f"{host} in {sorted(splits)}")


def _fixture_text(psai_root: Path, pilot_root: Path) -> str:
    """Join the names and text of every PSAI fixture outside the pilot.

    Args:
        psai_root: The PSAI fixture tree.
        pilot_root: The pilot directory to skip.

    Returns:
        File names and text file contents, one per line.
    """
    parts: list[str] = []
    for file in sorted(psai_root.rglob("*")):
        if not file.is_file() or pilot_root in file.parents:
            continue
        parts.append(file.name)
        if file.suffix in _TEXT_SUFFIXES:
            parts.append(file.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _source_ids(rows: list[Row]) -> Iterator[tuple[str, str]]:
    """Yield ``(case_id, unique_data_id)`` for every row and donor.

    Args:
        rows: Manifest rows that passed the schema.

    Yields:
        The case ID and one source ID it uses.
    """
    for row in rows:
        yield row["case_id"], row["unique_data_id"]
        if row["donor"] is not None:
            yield row["case_id"], row["donor"]["unique_data_id"]


def _check_fixture_overlap(rows: list[Row], pilot_root: Path) -> None:
    """Reject a source ID that already appears in ``tests/fixtures/psai``.

    Args:
        rows: Manifest rows that passed the schema.
        pilot_root: The pilot directory; its parent is scanned without it.

    Raises:
        ManifestError: With rule ``fixture_overlap`` when an ID is found.
    """
    root = pilot_root.resolve()
    text = _fixture_text(root.parent, root)
    for case_id, source_id in _source_ids(rows):
        if source_id in text:
            raise ManifestError(
                "fixture_overlap", f"{case_id} uses {source_id} from {root.parent}"
            )


def _check_gold_per_class(rows: list[Row]) -> None:
    """Require at least ``MIN_GOLD_PER_CLASS`` gold labels per class and split.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``gold_per_class`` when a class falls short.
    """
    for split in SPLITS:
        gold = Counter(row["gold_label"] for row in rows if row["split"] == split)
        for label in LABELS:
            if gold[label] < MIN_GOLD_PER_CLASS:
                raise ManifestError(
                    "gold_per_class", f"{split} has {gold[label]} {label}"
                )


def _check_human_decision(rows: list[Row]) -> None:
    """Require a human decision on final, swap and crop rows.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``human_decision`` when a decision is missing.
    """
    for row in rows:
        human = row["votes"]["human"]
        if row["label_source"] == "human" and human != row["gold_label"]:
            raise ManifestError(
                "human_decision", f"{row['case_id']} human vote {human} is not gold"
            )
        needs_human = row["split"] == "final" or row["row_kind"] in _HUMAN_KINDS
        if needs_human and row["label_source"] != "human":
            raise ManifestError(
                "human_decision", f"{row['case_id']} needs a human decision"
            )


def _agreement_gap(row: Row) -> str | None:
    """Describe why a non-human label lacks verified agreement.

    Args:
        row: One row whose ``label_source`` is not ``human``.

    Returns:
        The first gap found, or ``None`` when the votes agree.
    """
    votes: Mapping[str, Any] = row["votes"]
    gold = row["gold_label"]
    if votes["gemma_draft"] != gold:
        return f"Gemma draft {votes['gemma_draft']} is not {gold}"
    if votes["construction"] not in (None, gold):
        return f"construction {votes['construction']} is not {gold}"
    if votes[row["label_source"]] is None:
        return f"label source {row['label_source']} has no vote"
    qwen = votes["qwen_dom"]
    agree = 0 if qwen is None else qwen["run_labels"].count(gold)
    if agree < QWEN_MAJORITY:
        return f"Qwen agrees {agree}/5, need {QWEN_MAJORITY}"
    return None


def _check_verified_agreement(rows: list[Row]) -> None:
    """Accept a non-human label only when every model vote agrees with it.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``verified_agreement`` when a vote disagrees.
    """
    for row in rows:
        if row["label_source"] == "human":
            continue
        gap = _agreement_gap(row)
        if gap is not None:
            raise ManifestError("verified_agreement", f"{row['case_id']}: {gap}")


def _check_input_leak(rows: list[Row]) -> None:
    """Keep host and ``task_name`` text out of every model-input field.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``input_leak`` when a field holds that text.
    """
    for row in rows:
        secrets = [row["host"], row["task_name"]]
        if row["donor"] is not None:
            secrets.append(row["donor"]["host"])
        for field, value in row["model_input"].items():
            text = str(value).casefold()
            if any(secret.casefold() in text for secret in secrets):
                raise ManifestError(
                    "input_leak", f"{row['case_id']} {field} holds host or task text"
                )
