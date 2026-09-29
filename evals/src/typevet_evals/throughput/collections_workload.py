"""finvet collections workload: record mapping, seed questions and parity ([#236][i236]).

The finvet collections next-best-action split and its outcome seed stay in the
finvet checkout. This module reads them by path at run time and copies neither.
Only ``state`` plus the logged action, as ``proposed_action``, reaches the
judgment port. The record ``id``, ``meta`` and ``logged.outcome`` never do.

ECE follows finvet ``calibration.py``: equal-width bins, bin index
``min(int(p * n_bins), n_bins - 1)``, and the count-weighted gap divided by the
number of records with a probability. A failed record goes in no bin.

Attributes:
    JEV_DIR_ENV (str): Env var for the directory with ``val/train/test.jsonl``.
    QUESTIONS_ENV (str): Env var for the outcome seed file path.
    ENGAGED (str): Label of an engaged record.
    NOT_ENGAGED (str): Label of a record that did not engage.
    BASELINE_ECE (float): finvet #5 validation ECE.
    BASELINE_BASE_RATE (float): finvet #5 validation engaged base rate.
    PARITY_MAX_ECE (float): Highest validation ECE that keeps parity.
    COLLECTIONS_BASELINE (Baseline): The three finvet #5 values above.

Examples:
    ```python
    from typevet_evals.throughput.collections_workload import parity

    report = parity([(0.9, True), (0.1, False), (None, True)])
    assert report["scored"] == 2 and report["failures"] == 1
    ```

See Also:
    - [typevet.ports.judgment.JudgmentPort][]: the port each record reaches
    - [typevet_evals.throughput.public_workload][]: public records with a baseline

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from typevet.domain.judgment_questions import Choice, Noul
from typevet.ports.judgment import JudgmentPort

JEV_DIR_ENV: Final[str] = "TYPEVET_FINVET_JEV_DIR"
QUESTIONS_ENV: Final[str] = "TYPEVET_FINVET_QUESTIONS"
ENGAGED: Final[str] = "engaged"
NOT_ENGAGED: Final[str] = "not_engaged"
BASELINE_ECE: Final[float] = 0.1423
BASELINE_BASE_RATE: Final[float] = 0.515
PARITY_MAX_ECE: Final[float] = 0.1723
_NOUL_KEY: Final[str] = "will_engage"
_CHOICE_KEY: Final[str] = "accepted_offer"
_NO_OFFER: Final[str] = "NONE"


@dataclass(frozen=True, slots=True)
class CollectionsRecord:
    """One mapped collections record.

    Attributes:
        state (dict[str, Any]): Record ``state`` plus ``proposed_action``.
        label (str): ``engaged`` or ``not_engaged``.
        accepted_offer (str): Logged accepted offer, ``NONE`` when null.

    Examples:
        ```python
        row = CollectionsRecord(state={"a": 1}, label="engaged", accepted_offer="NONE")
        assert row.engaged
        ```
    """

    state: dict[str, Any]
    label: str
    accepted_offer: str

    @property
    def engaged(self) -> bool:
        """Return whether the logged outcome is engaged.

        Returns:
            True when ``label`` is ``engaged``.
        """
        return self.label == ENGAGED

    @property
    def positive(self) -> bool:
        """Return the positive label that parity scores.

        Returns:
            The same value as ``engaged``.
        """
        return self.engaged


@dataclass(frozen=True, slots=True)
class Baseline:
    """Reference values that one parity check compares against.

    Attributes:
        ece (float): Baseline ECE.
        base_rate (float): Baseline positive base rate.
        max_ece (float): Highest ECE that keeps parity.

    Examples:
        ```python
        Baseline(ece=0.17, base_rate=0.5, max_ece=0.20)
        ```
    """

    ece: float
    base_rate: float
    max_ece: float


COLLECTIONS_BASELINE: Final[Baseline] = Baseline(
    BASELINE_ECE, BASELINE_BASE_RATE, PARITY_MAX_ECE
)


@dataclass(frozen=True, slots=True)
class Bin:
    """One reliability bin.

    Attributes:
        lower (float): Inclusive lower edge.
        upper (float): Upper edge, inclusive only for the last bin.
        count (int): Number of probabilities in the bin.
        mean_prob (float): Mean probability, 0.0 when empty.
        frac_engaged (float): Fraction of engaged labels, 0.0 when empty.

    Examples:
        ```python
        b = Bin(lower=0.9, upper=1.0, count=1, mean_prob=1.0, frac_engaged=1.0)
        assert b.count == 1
        ```
    """

    lower: float
    upper: float
    count: int
    mean_prob: float
    frac_engaged: float


def map_record(record: Mapping[str, Any]) -> CollectionsRecord:
    """Map one finvet split record without changing it.

    Args:
        record: One parsed jsonl record.

    Returns:
        The state sent to the port and the logged labels.
    """
    logged = record["logged"]
    outcome = logged["outcome"]
    state = {**record["state"], "proposed_action": logged["action_taken"]}
    offer = outcome.get("accepted_offer")
    return CollectionsRecord(
        state=state,
        label=ENGAGED if outcome["engaged"] else NOT_ENGAGED,
        accepted_offer=_NO_OFFER if offer is None else str(offer),
    )


def load_records(path: Path) -> list[CollectionsRecord]:
    """Read and map a jsonl split in file order.

    Args:
        path: The jsonl split file.

    Returns:
        One mapped record for each non-blank line.
    """
    with path.open(encoding="utf-8") as fh:
        return [map_record(json.loads(line)) for line in fh if line.strip()]


def workload_paths(environ: Mapping[str, str], split: str = "val") -> tuple[Path, Path]:
    """Resolve the split file and the seed file from the environment.

    Args:
        environ: Environment mapping.
        split: Split name, for example ``val``, ``train`` or ``test``.

    Returns:
        The split jsonl path and the seed path.

    Raises:
        ValueError: When either env var is unset or empty.
    """
    for name in (JEV_DIR_ENV, QUESTIONS_ENV):
        if not environ.get(name):
            raise ValueError(f"{name} is not set")
    return Path(environ[JEV_DIR_ENV]) / f"{split}.jsonl", Path(environ[QUESTIONS_ENV])


def _seed_entry(
    seed: Mapping[str, Any], key: str, kind: str
) -> tuple[str, dict[str, Any]]:
    match seed.get(key):
        case {
            "type": str() as found,
            "instructions": str() as text,
            "criteria": dict() as criteria,
        } if found == kind:
            return text, criteria
        case _:
            msg = (
                f"seed {key!r} needs type {kind!r}, str instructions and dict criteria"
            )
            raise ValueError(msg)


def load_questions(path: Path) -> dict[str, Noul | Choice]:
    """Read the outcome seed into typed questions with verbatim text.

    Args:
        path: The seed JSON file.

    Returns:
        ``{"will_engage": Noul, "accepted_offer": Choice}``.

    Raises:
        ValueError: When a key is missing or has the wrong type or fields.
    """
    match json.loads(path.read_text(encoding="utf-8")):
        case dict() as seed:
            pass
        case _:
            raise ValueError("seed must be a JSON object")
    noul_text, noul_criteria = _seed_entry(seed, _NOUL_KEY, "noul")
    choice_text, choice_criteria = _seed_entry(seed, _CHOICE_KEY, "choice")
    return {
        _NOUL_KEY: Noul(instructions=noul_text, criteria=noul_criteria),
        _CHOICE_KEY: Choice(instructions=choice_text, criteria=choice_criteria),
    }


def judge_record(
    port: JudgmentPort,
    record: CollectionsRecord,
    questions: Mapping[str, Noul | Choice],
    model: str,
) -> float:
    """Send one record with both questions in one call.

    Args:
        port: The judgment port.
        record: The mapped record; only its dict state is sent.
        questions: Questions from `load_questions`.
        model: Backend model id.

    Returns:
        P(engaged), the ``will_engage`` Noul probability.
    """
    response = port.judge(record.state, questions, model)
    return response.nouls[_NOUL_KEY].noul


def reliability(
    probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> list[Bin]:
    """Bin probabilities against labels with equal-width bins.

    Args:
        probs: P(engaged) values in [0, 1].
        labels: True when the record engaged.
        n_bins: Number of bins.

    Returns:
        ``n_bins`` bins in ascending order.

    Raises:
        ValueError: When lengths differ, ``n_bins`` < 1 or a value is out of range.
    """
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels, strict=True):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p} is not in [0, 1]")
        buckets[min(int(p * n_bins), n_bins - 1)].append((p, y))
    bins = []
    for i, items in enumerate(buckets):
        n = len(items)
        mean = sum(p for p, _ in items) / n if n else 0.0
        frac = sum(1 for _, y in items if y) / n if n else 0.0
        bins.append(Bin(i / n_bins, (i + 1) / n_bins, n, mean, frac))
    return bins


def _ece_from_bins(bins: Sequence[Bin], total: int) -> float:
    if total == 0:
        return 0.0
    return sum(b.count * abs(b.mean_prob - b.frac_engaged) for b in bins) / total


def ece(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> float:
    """Return expected calibration error as finvet computes it.

    Args:
        probs: P(engaged) values in [0, 1].
        labels: True when the record engaged.
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]; 0.0 when there are no probabilities.
    """
    return _ece_from_bins(reliability(probs, labels, n_bins), len(probs))


def parity(
    rows: Sequence[tuple[float | None, bool]],
    baseline: Baseline | None = COLLECTIONS_BASELINE,
) -> dict[str, Any]:
    """Build the quality parity record for one set of answered records.

    Parity holds when every record has a probability and ECE is at most
    ``baseline.max_ece``. The default is the collections baseline (0.1423 +
    0.03).

    Args:
        rows: ``(P(positive) or None when the call failed, positive label)``.
        baseline: Reference values; ``None`` records the measures only.

    Returns:
        ECE, base rate over all rows, scored and failed counts, baseline
        values, delta, the bin table and ``meets_parity``. With no baseline,
        the baseline values, ``delta`` and ``meets_parity`` are ``None``.
    """
    scored = [(p, y) for p, y in rows if p is not None]
    probs = [p for p, _ in scored]
    labels = [y for _, y in scored]
    bins = reliability(probs, labels)
    value = _ece_from_bins(bins, len(scored))
    failures = len(rows) - len(scored)
    meets = None
    if baseline is not None:
        meets = bool(scored) and failures == 0 and value <= baseline.max_ece
    return {
        "ece": value,
        "base_rate": sum(1 for _, y in rows if y) / len(rows) if rows else 0.0,
        "scored": len(scored),
        "failures": failures,
        "baseline_ece": None if baseline is None else baseline.ece,
        "baseline_base_rate": None if baseline is None else baseline.base_rate,
        "delta": None if baseline is None else value - baseline.ece,
        "max_ece": None if baseline is None else baseline.max_ece,
        "meets_parity": meets,
        "bins": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_prob": b.mean_prob,
                "frac_engaged": b.frac_engaged,
            }
            for b in bins
        ],
    }
