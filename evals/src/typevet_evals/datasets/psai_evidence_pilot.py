"""Manifest loader for the PSAI claim-about-screen evidence pilot v1 ([#192][i192]).

The pilot asks whether one short claim about a computer-use screenshot is
``supported``, ``contradicted`` or ``insufficient_evidence``. The design lives
in [#189 revision 4][r4]. This module loads its manifest offline. It checks
the JSON schema first, then the cross-row rules below, and raises one
``ManifestError`` that names the first rule that fails.

| Rule | Check |
|---|---|
| ``schema`` | The manifest matches ``schema.json``. |
| ``case_id_unique`` | No ``case_id`` is used twice. |
| ``split_counts`` | Each split holds 8 baseline, 8 claim_axis, 4 swap, 4 crop. |
| ``split_isolation`` | A host or donor belongs to one split only. |
| ``fixture_overlap`` | No ``unique_data_id`` is in other PSAI fixtures. |
| ``rejected_items`` | A rejected ID is unique, is not a row ID, is not in PSAI fixtures. |
| ``task_inventory`` | 8 tasks per split, each with one baseline, claim_axis, image row. |
| ``gold_per_class`` | Each split holds at least 4 gold per class. |
| ``human_decision`` | Final, swap and crop rows carry a human decision. |
| ``verified_agreement`` | A non-human label has Gemma, construction and Qwen agreement. |
| ``input_leak`` | No host, ``task_name`` or 4-word ``task_name`` run is in a model-input field. |
| ``gold_label_leak`` | No model-input field holds the gold label, joined by space, _ or a dash. |

Rejected items sit in their own list with reason codes. They are not rows and
no count rule counts them. A task is one ``unique_data_id`` in one split.

[i192]: https://github.com/Alberto-Codes/typevet/issues/192
[r4]: https://github.com/Alberto-Codes/typevet/issues/189#issuecomment-5875661660

Examples:
    Load the synthetic manifest:

    ```python
    from pathlib import Path

    from typevet_evals.datasets.psai_evidence_pilot import load_manifest

    root = Path("tests/fixtures/psai/evidence_pilot_v1")
    manifest = load_manifest(root / "synthetic" / "manifest.json")
    assert len(manifest.rows) == 72
    ```

See Also:
    - [typevet_evals.datasets.psai_vision][]: earlier PSAI screenshot fixtures
    - tests/fixtures/psai/evidence_pilot_v1/schema.json: the manifest schema

Attributes:
    PILOT_DIR_NAME (str): Directory name that marks the pilot root.
    SPLITS (tuple[str, ...]): Split names in order.
    LABELS (tuple[str, ...]): Gold label classes.
    SPLIT_KIND_COUNTS (dict[str, int]): Required rows per kind in each split.
    MIN_GOLD_PER_CLASS (int): Smallest gold count per class in each split.
    QWEN_MAJORITY (int): Qwen runs that must agree with a non-human label.
    TASKS_PER_SPLIT (int): Distinct ``unique_data_id`` values in each split.
    TASK_RUN_WORDS (int): Consecutive ``task_name`` words that make an input leak.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
TASKS_PER_SPLIT: Final = 8
TASK_RUN_WORDS: Final = 4

_TEXT_SUFFIXES: Final = frozenset({".json", ".jsonl", ".txt", ".csv", ".md"})
_HUMAN_KINDS: Final = frozenset({"swap", "crop"})
_TASK_ROLES: Final = {"baseline": 1, "claim_axis": 1, "image": 1}
_WORD: Final = re.compile(r"[^\W_]+")

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

    The rules run in the order of the module rule table. The PSAI fixture
    tree is read once and shared by ``fixture_overlap`` and ``rejected_items``.

    Args:
        path: The manifest JSON file.
        pilot_root: The ``evidence_pilot_v1`` directory that holds
            ``schema.json``. Its parent is the PSAI fixture tree that the
            ``fixture_overlap`` and ``rejected_items`` rules scan. Defaults
            to the nearest ancestor of ``path`` with that name.

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
    rejected: list[Row] = data["rejected"]
    _check_case_id_unique(rows)
    _check_split_counts(rows)
    _check_split_isolation(rows)
    fixture_text = _fixture_text(root.resolve().parent, root.resolve())
    _check_fixture_overlap(rows, fixture_text)
    _check_rejected_items(rows, rejected, fixture_text)
    _check_task_inventory(rows)
    _check_gold_per_class(rows)
    _check_human_decision(rows)
    _check_verified_agreement(rows)
    _check_input_leak(rows)
    _check_gold_label_leak(rows)
    return PilotManifest(rows=tuple(rows), rejected=tuple(rejected))


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


def _check_case_id_unique(rows: list[Row]) -> None:
    """Require each ``case_id`` to name one row only.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``case_id_unique`` when a case ID repeats.
    """
    counts = Counter(row["case_id"] for row in rows)
    repeated = sorted(case_id for case_id, count in counts.items() if count > 1)
    if repeated:
        raise ManifestError("case_id_unique", f"{repeated[0]} is used twice")


def _task_roles(rows: list[Row], split: str) -> dict[str, dict[str, int]]:
    """Count the role of each row per task in one split.

    Args:
        rows: Manifest rows that passed the schema.
        split: One of ``SPLITS``.

    Returns:
        Role counts keyed by ``unique_data_id``. A ``swap`` or ``crop`` row
        counts as role ``image``.
    """
    tasks: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["split"] == split:
            kind = row["row_kind"]
            tasks[row["unique_data_id"]]["image" if kind in _HUMAN_KINDS else kind] += 1
    return {source_id: dict(roles) for source_id, roles in tasks.items()}


def _check_task_inventory(rows: list[Row]) -> None:
    """Require 8 tasks per split, each with one row of each role.

    A task is one ``unique_data_id`` in one split. Its roles are one
    ``baseline``, one ``claim_axis`` and one image row (``swap`` or ``crop``).

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``task_inventory`` when a split has another
            task count or a task has another role mix.
    """
    for split in SPLITS:
        tasks = _task_roles(rows, split)
        if len(tasks) != TASKS_PER_SPLIT:
            raise ManifestError(
                "task_inventory",
                f"{split} has {len(tasks)} tasks, need {TASKS_PER_SPLIT}",
            )
        for source_id, roles in sorted(tasks.items()):
            if roles != _TASK_ROLES:
                raise ManifestError(
                    "task_inventory", f"{split} {source_id} has {roles}"
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


def _check_fixture_overlap(rows: list[Row], fixture_text: str) -> None:
    """Reject a source ID that already appears in ``tests/fixtures/psai``.

    Args:
        rows: Manifest rows that passed the schema.
        fixture_text: PSAI fixture names and text outside the pilot, from
            ``_fixture_text``.

    Raises:
        ManifestError: With rule ``fixture_overlap`` when an ID is found.
    """
    for case_id, source_id in _source_ids(rows):
        if source_id in fixture_text:
            raise ManifestError(
                "fixture_overlap", f"{case_id} uses {source_id} from PSAI fixtures"
            )


def _check_rejected_items(
    rows: list[Row], rejected: list[Row], fixture_text: str
) -> None:
    """Keep each rejected ID apart from rows, other rejects and PSAI fixtures.

    Args:
        rows: Manifest rows that passed the schema.
        rejected: Rejected items that passed the schema.
        fixture_text: PSAI fixture names and text outside the pilot, from
            ``_fixture_text``.

    Raises:
        ManifestError: With rule ``rejected_items`` when a rejected ID is a
            row ID, repeats in ``rejected`` or is in the PSAI fixtures.
    """
    row_ids = {row["unique_data_id"] for row in rows}
    seen: set[str] = set()
    for item in rejected:
        source_id = item["unique_data_id"]
        if source_id in row_ids:
            raise ManifestError("rejected_items", f"rejected {source_id} is a row ID")
        if source_id in seen:
            raise ManifestError("rejected_items", f"rejected {source_id} appears twice")
        if source_id in fixture_text:
            raise ManifestError(
                "rejected_items", f"rejected {source_id} is in PSAI fixtures"
            )
        seen.add(source_id)


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


def _word_runs(text: str, size: int) -> set[tuple[str, ...]]:
    """Return each run of ``size`` consecutive words in case-folded ``text``.

    A word is a run of letters or digits; ``_`` and punctuation split words.

    Args:
        text: The text to split.
        size: Words per run.

    Returns:
        The word runs, as tuples.
    """
    words = _WORD.findall(text.casefold())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def _holds_task_run(text: str, task_name: str) -> bool:
    """Tell whether ``text`` copies ``TASK_RUN_WORDS`` words of ``task_name``.

    A ``task_name`` with fewer words must be copied whole.

    Args:
        text: One model-input field value.
        task_name: The row's task name.

    Returns:
        ``True`` when the text holds such a run of task words.
    """
    size = min(TASK_RUN_WORDS, len(_WORD.findall(task_name)))
    if size == 0:
        return False
    return not _word_runs(task_name, size).isdisjoint(_word_runs(text, size))


def _check_input_leak(rows: list[Row]) -> None:
    """Keep host and ``task_name`` text out of every model-input field.

    A field also leaks when it holds ``TASK_RUN_WORDS`` consecutive words of
    ``task_name``, or the whole name when it has fewer words.

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
            if any(secret.casefold() in text for secret in secrets) or (
                _holds_task_run(text, row["task_name"])
            ):
                raise ManifestError(
                    "input_leak", f"{row['case_id']} {field} holds host or task text"
                )


def _gold_pattern(label: str) -> re.Pattern[str]:
    """Match ``label`` with no letter or digit on either side.

    The words of the label may be joined by any run of spaces, ``_``, ``-``
    or the Unicode dashes U+2010 to U+2015. A ``_`` next to the label does
    not hide it, so ``supported_by`` matches and ``unsupported`` does not.

    Args:
        label: One of ``LABELS``.

    Returns:
        A pattern for the label, to search in case-folded text.
    """
    body = r"[\s_\u2010-\u2015-]+".join(re.escape(part) for part in label.split("_"))
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])")


def _check_gold_label_leak(rows: list[Row]) -> None:
    """Keep each row's gold label out of its model-input fields.

    Field text is NFKC-normalized and soft hyphens (U+00AD) are removed
    before the match, so fullwidth or rich-text forms do not hide the label.

    Args:
        rows: Manifest rows that passed the schema.

    Raises:
        ManifestError: With rule ``gold_label_leak`` when a field holds the
            gold label as ``_gold_pattern`` matches it.
    """
    for row in rows:
        pattern = _gold_pattern(row["gold_label"])
        for field, value in row["model_input"].items():
            text = unicodedata.normalize("NFKC", str(value)).replace("\u00ad", "")
            if pattern.search(text.casefold()):
                raise ManifestError(
                    "gold_label_leak", f"{row['case_id']} {field} holds its gold label"
                )
