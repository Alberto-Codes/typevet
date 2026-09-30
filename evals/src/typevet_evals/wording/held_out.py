"""Score seed and evolved wording on the held-out rows and build receipts (#309).

``score_held_out`` asks one judgevet ``SystemOnePort`` the seed wording and
then the evolved wording about each held-out row, once each. It refuses a
record whose ``split`` is not ``test`` before any call, and it stops at the
first failure and records it. ``held_out_receipt`` turns the scored pairs
into the metrics of both wordings, the paired bootstrap intervals (context
only), the pre-registered verdict, the #133 re-measurement and each call's
latency and input tokens with their totals (#327).

``stratified_subset`` picks the fixed validation rows the evolution selects
on, and ``evolution_artifact`` records what one evolution run produced.

This module does not import judgevet: the caller gives the port and the seed
``Noul``, as for the #306 transport.

Attributes:
    DEFAULT_TRAIN_ROWS (int): Train rows the evolution reflects on by default.
    HELD_OUT_SPLIT (str): The ``split`` every held-out record carries.
    VALIDATION_SPLIT (str): The ``split`` every smoke record carries (#329).
    REFERENCE_ECE (float): The #133 DIFrauD ECE the seed wording is compared with.

Examples:
    ```python
    from typevet_evals.wording.held_out import held_out_receipt, score_held_out

    run = score_held_out(
        port,
        seed_noul,
        "is_scam",
        evolved_text=evolved,
        rows=HeldOutRows(splits.held_out, splits.prior_measured_ids),
        judge_model=model,
        failures=(ProviderError,),
    )
    receipt = held_out_receipt(
        run,
        seed_text=seed_text,
        evolved_text=evolved,
        backend="vllm",
        model=model,
        pins=pins,
        identity=identity,
    )
    ```

See Also:
    - [typevet_evals.wording.metrics][]: metrics, bootstrap and pass rule
    - [typevet_evals.wording.runner][]: the evolution runner
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet_evals.datasets.difraud import DIFrauDRecord
from typevet_evals.wording.calls import CallRecord, call_summary
from typevet_evals.wording.metrics import (
    paired_bootstrap,
    pass_verdict,
    wording_metrics,
)
from typevet_evals.wording.runner import POSITIVE_LABEL, WordingRun, WordingRunConfig
from typevet_evals.wording.transport import JudgePort, SeedNoul

DEFAULT_TRAIN_ROWS: Final[int] = 1000
HELD_OUT_SPLIT: Final[str] = "test"
VALIDATION_SPLIT: Final[str] = "validation"
REFERENCE_ECE: Final[float] = 0.158


def _rank(record: DIFrauDRecord, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{record.record_id}".encode()).digest()


def _quotas(counts: Mapping[str, int], size: int) -> dict[str, int]:
    total = sum(counts.values())
    shares = {label: size * n / total for label, n in counts.items()}
    quotas = {label: math.floor(share) for label, share in shares.items()}
    left = size - sum(quotas.values())
    by_remainder = sorted(
        shares, key=lambda label: (quotas[label] - shares[label], label)
    )
    for label in by_remainder[:left]:
        quotas[label] += 1
    return quotas


def stratified_subset(
    records: Iterable[DIFrauDRecord], size: int, *, seed: int = 0
) -> tuple[DIFrauDRecord, ...]:
    """Return ``size`` records with the label shares of ``records``.

    Records are ordered by ``sha256(f"{seed}:{record_id}")``, so the subset
    depends on the seed and the record ids, not on the input order. Each
    label gets ``floor`` of its share of ``size``; the rows left go to the
    labels with the largest remainders.

    Args:
        records: The records to choose from.
        size: Rows to keep.
        seed: The ordering seed.

    Returns:
        The chosen records in hash order.

    Raises:
        ValueError: When ``size`` is below 1 or above the record count.
    """
    ordered = sorted(records, key=lambda record: _rank(record, seed))
    if not 1 <= size <= len(ordered):
        msg = f"size {size} is not between 1 and {len(ordered)}"
        raise ValueError(msg)
    counts: dict[str, int] = {}
    for record in ordered:
        counts[record.example.label] = counts.get(record.example.label, 0) + 1
    quotas = _quotas(counts, size)
    chosen: list[DIFrauDRecord] = []
    for record in ordered:
        if quotas[record.example.label] > 0:
            quotas[record.example.label] -= 1
            chosen.append(record)
    return tuple(chosen)


def train_subset(
    records: Sequence[DIFrauDRecord],
    rows: int | None = DEFAULT_TRAIN_ROWS,
    *,
    seed: int = 0,
) -> tuple[DIFrauDRecord, ...]:
    """Return the train rows the evolution reflects on.

    Args:
        records: The train records.
        rows: Rows to keep, as a ``stratified_subset``; None keeps every row.
        seed: The ordering seed.

    Returns:
        The stratified subset, or every record in input order when ``rows``
        is None.
    """
    if rows is None:
        return tuple(records)
    return stratified_subset(records, rows, seed=seed)


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """Both wordings' scam probability for one held-out row.

    Attributes:
        record_id (str): The DIFrauD record id.
        label (int): 1 for scam, 0 for legit.
        seed_probability (float): The seed wording's answer.
        evolved_probability (float): The evolved wording's answer.

    Examples:
        ```python
        ScoredPair("sha256:0123", 1, 0.6, 0.9)
        ```
    """

    record_id: str
    label: int
    seed_probability: float
    evolved_probability: float


@dataclass(frozen=True, slots=True)
class HeldOutRun:
    """The scored held-out pairs and the call counts of each wording.

    Attributes:
        pairs (tuple[ScoredPair, ...]): Rows both wordings answered, in order.
        seed_calls (int): Port calls made with the seed wording, including a
            failed one.
        evolved_calls (int): Port calls made with the evolved wording,
            including a failed one.
        stopped (str | None): The first failure as ``Type: message``, or None.
        call_records (tuple[CallRecord, ...]): Per-call latency and input
            tokens, for example ``TimedJudgePort.records`` (#327); empty by
            default.

    Examples:
        ```python
        run = score_held_out(port, seed, "is_scam", evolved_text=text, ...)
        # records, judge_model and failures are required too
        assert run.stopped is None
        ```
    """

    pairs: tuple[ScoredPair, ...]
    seed_calls: int
    evolved_calls: int
    stopped: str | None
    call_records: tuple[CallRecord, ...] = ()

    @property
    def calls(self) -> int:
        """Return the port calls of both wordings.

        Returns:
            ``seed_calls + evolved_calls``.
        """
        return self.seed_calls + self.evolved_calls


@dataclass(frozen=True, slots=True)
class HeldOutRows:
    """The held-out records and the record ids that must never be scored.

    Construction refuses a record whose ``split`` is not ``test`` or whose id
    is excluded, so a bad row fails before any port call.

    Attributes:
        records (tuple[DIFrauDRecord, ...]): The held-out records, in order.
        excluded_ids (frozenset[str]): Ids that must never be scored, such as
            ``DIFrauDSplits.prior_measured_ids`` (the #236 rows, which also
            carry split ``test``).

    Examples:
        ```python
        rows = HeldOutRows(splits.held_out, splits.prior_measured_ids)
        ```
    """

    records: tuple[DIFrauDRecord, ...]
    excluded_ids: frozenset[str]

    def __post_init__(self) -> None:
        """Refuse a non-held-out or excluded record.

        Raises:
            ValueError: If a record's ``split`` is not ``test`` or its id is in
                ``excluded_ids``.
        """
        for record in self.records:
            if record.example.split != HELD_OUT_SPLIT:
                msg = f"held-out rows hold a {record.example.split!r} record"
                raise ValueError(msg)
            if record.record_id in self.excluded_ids:
                msg = f"held-out rows hold excluded (#236) record {record.record_id}"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ValidationRows:
    """Validation records for a smoke run of the held-out loop (#329).

    Construction refuses a record whose ``split`` is not ``validation``, so a
    smoke cannot score a held-out row.

    Attributes:
        records (tuple[DIFrauDRecord, ...]): The validation records, in order.

    Examples:
        ```python
        rows = ValidationRows(stratified_subset(splits.validation, 6))
        ```
    """

    records: tuple[DIFrauDRecord, ...]

    def __post_init__(self) -> None:
        """Refuse a record that is not a validation record.

        Raises:
            ValueError: If a record's ``split`` is not ``validation``.
        """
        for record in self.records:
            if record.example.split != VALIDATION_SPLIT:
                msg = f"validation rows hold a {record.example.split!r} record"
                raise ValueError(msg)


def score_held_out(
    port: JudgePort,
    seed: SeedNoul,
    key: str,
    *,
    evolved_text: str,
    rows: HeldOutRows | ValidationRows,
    judge_model: str,
    failures: tuple[type[Exception], ...],
) -> HeldOutRun:
    """Ask the seed wording, then the evolved wording, about each held-out row.

    Args:
        port: The judgevet ``SystemOnePort``.
        seed: The seed ``Noul``; its criteria go with both wordings.
        key: The question name.
        evolved_text: The evolved wording.
        rows: The checked held-out records, or checked validation records
            for a smoke run.
        judge_model: The model name sent to the port.
        failures: The exception types that count as a backend failure, for
            example judgevet's ``ProviderError``. Any other exception propagates.

    Returns:
        The pairs both wordings answered before any failure, the call counts
        of each wording, counted as each call starts, and the first failure.
    """
    arms = (("seed", str(seed.instructions)), ("evolved", evolved_text))
    pairs: list[ScoredPair] = []
    calls = {"seed": 0, "evolved": 0}
    for record in rows.records:
        answers: list[float] = []
        for arm, text in arms:
            noul = type(seed)(instructions=text, criteria=seed.criteria)
            calls[arm] += 1
            try:
                response = port.system_one(
                    record.example.text, {key: noul}, judge_model
                )
            except failures as exc:
                stopped = f"{type(exc).__name__}: {exc}"
                return HeldOutRun(
                    tuple(pairs), calls["seed"], calls["evolved"], stopped
                )
            answers.append(float(response.nouls[key].noul))
        label = int(record.example.label == POSITIVE_LABEL)
        pairs.append(ScoredPair(record.record_id, label, answers[0], answers[1]))
    return HeldOutRun(tuple(pairs), calls["seed"], calls["evolved"], None)


def held_out_receipt(
    run: HeldOutRun,
    *,
    seed_text: str,
    evolved_text: str,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the key-free held-out receipt.

    Metrics, bootstrap, verdict and the #133 comparison are None when the run
    stopped or scored no row: a partial run gives no verdict. ``per_call``
    and ``call_summary`` come from ``run.call_records``.

    Args:
        run: The scored held-out run.
        seed_text: The seed wording.
        evolved_text: The evolved wording.
        backend: ``llama_cpp`` or ``vllm``.
        model: The served model id.
        pins: Dataset, server and weights pins.
        identity: The experiment identity mapping.

    Returns:
        A JSON-serializable receipt.
    """
    labels = [p.label for p in run.pairs]
    seed_p = [p.seed_probability for p in run.pairs]
    evolved_p = [p.evolved_probability for p in run.pairs]
    complete = run.stopped is None and bool(run.pairs)
    metrics = bootstrap = verdict = reference = None
    if complete:
        seed = wording_metrics(seed_p, labels)
        evolved = wording_metrics(evolved_p, labels)
        metrics = {"seed": seed.to_mapping(), "evolved": evolved.to_mapping()}
        bootstrap = paired_bootstrap(seed_p, evolved_p, labels).to_mapping()
        verdict = pass_verdict(seed, evolved).to_mapping()
        reference = {
            "prior_ece": REFERENCE_ECE,
            "seed_ece": seed.ece,
            "difference": seed.ece - REFERENCE_ECE,
        }
    return {
        "issue": 309,
        "backend": backend,
        "model": model,
        "seed_text": seed_text,
        "evolved_text": evolved_text,
        "rows": len(run.pairs),
        "calls": {
            "seed": run.seed_calls,
            "evolved": run.evolved_calls,
            "total": run.calls,
        },
        "stopped": run.stopped,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "verdict": verdict,
        "reference_133": reference,
        "pairs": [
            {
                "record_id": p.record_id,
                "label": p.label,
                "seed": p.seed_probability,
                "evolved": p.evolved_probability,
            }
            for p in run.pairs
        ],
        "per_call": [r.to_mapping() for r in run.call_records],
        "call_summary": call_summary(run.call_records),
        "pins": dict(pins),
        "identity": dict(identity),
    }


def evolution_artifact(
    run: WordingRun,
    *,
    config: WordingRunConfig,
    train: Sequence[DIFrauDRecord],
    validation: Sequence[DIFrauDRecord],
    call_records: Sequence[CallRecord] = (),
) -> dict[str, Any]:
    """Record the seed and evolved wording and how they were chosen.

    Args:
        run: The evolution run.
        config: The run settings; a ``BaseLlm`` reflector is recorded by its
            ``model`` name.
        train: The train records gepa-adk reflected on.
        validation: The validation records gepa-adk selected on.
        call_records: Per-call latency and input tokens of the judge, for
            example ``TimedJudgePort.records`` (#327).

    Returns:
        A JSON-serializable artifact with both texts, the settings, the
        validation record ids, gepa-adk's result and the judge calls.
    """
    reflector = config.reflector
    return {
        "issue": 309,
        "seed_text": run.seed_text,
        "evolved_text": run.evolved_text,
        "reflector": reflector if isinstance(reflector, str) else reflector.model,
        "judge_model": config.judge_model,
        "train_rows": len(train),
        "validation_ids": [r.record_id for r in validation],
        "length_cap": math.floor(config.length_ratio * len(run.seed_text)),
        "settings": {
            "max_iterations": config.max_iterations,
            "patience": config.patience,
            "reflection_max_trials": config.reflection_max_trials,
            "reflection_minibatch_size": config.reflection_minibatch_size,
            "max_concurrent_evals": config.max_concurrent_evals,
            "seed": config.seed,
            "length_ratio": config.length_ratio,
        },
        "result": run.result.to_dict(),
        "per_call": [r.to_mapping() for r in call_records],
        "call_summary": call_summary(call_records),
    }
