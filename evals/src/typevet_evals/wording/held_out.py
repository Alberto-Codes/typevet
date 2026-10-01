"""Score seed and evolved wording on the held-out rows and build receipts (#309).

``score_held_out`` asks one judgevet ``SystemOnePort`` the seed wording and
then the evolved wording about each held-out row, once each. It refuses a
record whose ``split`` is not ``test`` before any call, and it stops at the
first failure and records it. The run names the ``split`` of its row type.
``held_out_receipt`` refuses a run that is not ``test`` (#339) and turns the
scored pairs into the metrics of both wordings, the paired bootstrap intervals (context
only), the pre-registered verdict, the #133 re-measurement and each call's
latency and input tokens with their totals (#327).
The receipt pins both wordings by digest in ``wording_digests`` (#362).
It records the evolved selection and both full part mappings (#363).
``score_held_out`` takes the run's ``WordingParts``, so the evolved arm
sends each evolved part, criteria included (#372).
A ``Choice`` seed (#369) records each row's two label distributions as a
``ChoicePair``. Its receipt holds accuracy, Brier score, ECE and Cohen's
kappa, the same paired bootstrap and pass rule, and no #133 reference.
Both receipts record the seed's ``part_table``.

``stratified_subset`` picks the fixed validation rows the evolution selects
on, and ``evolution_artifact`` records what one evolution run produced.

This module does not import judgevet: the caller gives the port and the seed
``Noul`` or ``Choice``, as for the #306 transport.

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
        evolved=WordingParts.instructions_only(seed_noul, evolved),
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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from typevet_evals.datasets.difraud import DIFrauDRecord
from typevet_evals.wording.calls import CallRecord, call_summary
from typevet_evals.wording.choice import (
    ChoicePair,
    choice_arm_metrics,
    choice_bootstrap,
)
from typevet_evals.wording.digests import wording_fields
from typevet_evals.wording.metrics import (
    paired_bootstrap,
    pass_verdict,
    wording_metrics,
)
from typevet_evals.wording.parts import (
    SeedNoul,
    WordingParts,
    part_table,
    question_from_parts,
    receipt_parts,
    seed_mapping,
)
from typevet_evals.wording.rows import AnyRow, as_row
from typevet_evals.wording.runner import (
    WordingRun,
    WordingRunConfig,
    part_caps,
    scorer_for,
)
from typevet_evals.wording.transport import JudgePort, answer_body

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
        split (str): The ``split`` of the scored rows, keyword only:
            ``test`` for ``HeldOutRows``, ``validation`` for
            ``ValidationRows`` (#339).
        parts (WordingParts | None): The parts the caller scored (#363),
            keyword only; ``score_held_out`` always sets them. A receipt
            records them; None records the ``instructions`` part only.
        choice_pairs (tuple[ChoicePair, ...]): The pairs of a ``Choice``
            seed (#369), keyword only; ``pairs`` is then empty.
        labels (tuple[str, ...]): The ``Choice`` labels, keyword only; empty
            for a ``Noul`` seed.
        part_table (Mapping[str, str | int]): The seed's ``part_table``,
            keyword only; ``score_held_out`` always sets it.

    Examples:
        ```python
        run = score_held_out(port, seed, "is_scam", evolved=parts, ...)
        # records, judge_model and failures are required too
        assert run.stopped is None
        ```
    """

    pairs: tuple[ScoredPair, ...]
    seed_calls: int
    evolved_calls: int
    stopped: str | None
    call_records: tuple[CallRecord, ...] = ()
    split: str = field(kw_only=True)
    parts: WordingParts | None = field(default=None, kw_only=True)
    choice_pairs: tuple[ChoicePair, ...] = field(default=(), kw_only=True)
    labels: tuple[str, ...] = field(default=(), kw_only=True)
    part_table: Mapping[str, str | int] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        """Refuse an unknown split.

        Raises:
            ValueError: If ``split`` is not ``test`` or ``validation``.
        """
        if self.split not in (HELD_OUT_SPLIT, VALIDATION_SPLIT):
            msg = f"split {self.split!r} is not {HELD_OUT_SPLIT!r} or {VALIDATION_SPLIT!r}"
            raise ValueError(msg)

    @property
    def calls(self) -> int:
        """Return the port calls of both wordings.

        Returns:
            ``seed_calls + evolved_calls``.
        """
        return self.seed_calls + self.evolved_calls

    @property
    def rows(self) -> int:
        """Return the rows both wordings answered.

        Returns:
            The count of ``pairs`` or of ``choice_pairs``.
        """
        return len(self.pairs) + len(self.choice_pairs)


@dataclass(frozen=True, slots=True)
class HeldOutRows:
    """The held-out records and the record ids that must never be scored.

    Construction refuses a record whose ``split`` is not ``test`` or whose id
    is excluded, so a bad row fails before any port call.

    Attributes:
        records (tuple[AnyRow, ...]): The held-out DIFrauD records or
            ``WordingRow`` values, in order.
        excluded_ids (frozenset[str]): Ids that must never be scored, such as
            ``DIFrauDSplits.prior_measured_ids`` (the #236 rows, which also
            carry split ``test``).
        split (ClassVar[str]): ``test``, the split a run of these rows names.

    Examples:
        ```python
        rows = HeldOutRows(splits.held_out, splits.prior_measured_ids)
        ```
    """

    split: ClassVar[str] = HELD_OUT_SPLIT
    records: tuple[AnyRow, ...]
    excluded_ids: frozenset[str]

    def __post_init__(self) -> None:
        """Refuse a non-held-out or excluded record or row, read through ``as_row``.

        Raises:
            ValueError: If a record's ``split`` is not ``test`` or its id is in
                ``excluded_ids``.
        """
        for record in map(as_row, self.records):
            if record.split != HELD_OUT_SPLIT:
                msg = f"held-out rows hold a {record.split!r} record"
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
        records (tuple[AnyRow, ...]): The validation records or rows, in order.
        split (ClassVar[str]): ``validation``, the split a run of these rows
            names.

    Examples:
        ```python
        rows = ValidationRows(stratified_subset(splits.validation, 6))
        ```
    """

    split: ClassVar[str] = VALIDATION_SPLIT
    records: tuple[AnyRow, ...]

    def __post_init__(self) -> None:
        """Refuse a record or row that is not a validation one, through ``as_row``.

        Raises:
            ValueError: If a record's ``split`` is not ``validation``.
        """
        for record in map(as_row, self.records):
            if record.split != VALIDATION_SPLIT:
                msg = f"validation rows hold a {record.split!r} record"
                raise ValueError(msg)


def _check_golds(rows: HeldOutRows | ValidationRows, golds: Sequence[str]) -> None:
    """Refuse a row whose gold label the seed cannot score, before any call.

    Args:
        rows: The checked rows.
        golds: The gold labels the seed's scorer accepts.

    Raises:
        ValueError: If a gold label is not in ``golds``; the message names
            the record id only.
    """
    for row in map(as_row, rows.records):
        if row.gold not in golds:
            msg = f"record {row.record_id!r} has a gold label outside the seed"
            raise ValueError(msg)


def _pair(record_id: str, gold: str, answers: list[Any]) -> ScoredPair | ChoicePair:
    """Return the pair of one row from both arms' transport bodies.

    Args:
        record_id: The record id.
        gold: The gold label.
        answers: The seed body and the evolved body.

    Returns:
        A ``ChoicePair`` for ``probabilities`` bodies, else a ``ScoredPair``.
    """
    seed, evolved = answers
    if "probabilities" in seed:
        return ChoicePair(
            record_id, gold, seed["probabilities"], evolved["probabilities"]
        )
    p_seed, p_evolved = float(seed["probability"]), float(evolved["probability"])
    return ScoredPair(record_id, int(gold == "1"), p_seed, p_evolved)


def score_held_out(
    port: JudgePort,
    seed: SeedNoul,
    question_name: str,
    *,
    evolved: WordingParts,
    rows: HeldOutRows | ValidationRows,
    judge_model: str,
    failures: tuple[type[Exception], ...],
) -> HeldOutRun:
    """Ask the seed wording, then the evolved wording, about each held-out row.

    Args:
        port: The judgevet ``SystemOnePort``.
        seed: The seed ``Noul`` or ``Choice``; its parts make the seed
            question, and its type picks the answer it reads (#369).
        question_name: The question name sent to the port.
        evolved: The run's ``WordingParts`` (#363); the evolved arm sends
            every evolved part. ``WordingParts.instructions_only`` gives the
            parts of an evolved ``instructions`` text, so both arms send the
            seed criteria.
        rows: The checked held-out records, or checked validation records
            for a smoke run.
        judge_model: The model name sent to the port.
        failures: The exception types that count as a backend failure, for
            example judgevet's ``ProviderError``. Any other exception propagates.

    Returns:
        The pairs both wordings answered before any failure, the call counts
        of each wording, counted as each call starts, the first failure, the
        ``split`` of the row type, the given ``WordingParts``, the ``Choice``
        labels and the seed's ``part_table``.

    Raises:
        ValueError: Before any call, when the seed is a ``Score``, the given
            parts do not hold the seed's parts, or a gold label is outside
            the seed.
    """
    if dict(evolved.seed) != seed_mapping(seed):
        raise ValueError("the parts do not hold the seed's parts")
    _check_golds(rows, scorer_for(seed)[1])
    kind = type(seed).__name__
    arms = (
        ("seed", question_from_parts(seed, evolved.seed)),
        ("evolved", question_from_parts(seed, evolved.evolved)),
    )
    pairs: list[ScoredPair | ChoicePair] = []
    calls = {"seed": 0, "evolved": 0}
    stopped = None
    for row in map(as_row, rows.records):
        answers: list[Any] = []
        for arm, question in arms:
            calls[arm] += 1
            try:
                response = port.system_one(
                    row.state, {question_name: question}, judge_model
                )
            except failures as exc:
                stopped = f"{type(exc).__name__}: {exc}"
                break
            answers.append(answer_body(response, question_name, kind))
        if stopped is not None:
            break
        pairs.append(_pair(row.record_id, row.gold, answers))
    return HeldOutRun(
        tuple(p for p in pairs if isinstance(p, ScoredPair)),
        calls["seed"],
        calls["evolved"],
        stopped,
        split=rows.split,
        parts=evolved,
        choice_pairs=tuple(p for p in pairs if isinstance(p, ChoicePair)),
        labels=scorer_for(seed)[1] if kind == "Choice" else (),
        part_table=part_table(seed),
    )


def _measures(run: HeldOutRun) -> tuple[Any, Any, Any, Any]:
    """Return the metrics, bootstrap, verdict and #133 reference of a run.

    Args:
        run: A complete held-out run.

    Returns:
        Four JSON mappings; the reference is None for a ``Choice`` run.
    """
    if run.labels:
        seed_c, evolved_c = choice_arm_metrics(run.choice_pairs, run.labels)
        return (
            {"seed": seed_c.to_mapping(), "evolved": evolved_c.to_mapping()},
            choice_bootstrap(run.choice_pairs, run.labels).to_mapping(),
            pass_verdict(seed_c, evolved_c).to_mapping(),
            None,
        )
    labels = [p.label for p in run.pairs]
    seed_p = [p.seed_probability for p in run.pairs]
    evolved_p = [p.evolved_probability for p in run.pairs]
    seed = wording_metrics(seed_p, labels)
    evolved = wording_metrics(evolved_p, labels)
    reference = {
        "prior_ece": REFERENCE_ECE,
        "seed_ece": seed.ece,
        "difference": seed.ece - REFERENCE_ECE,
    }
    return (
        {"seed": seed.to_mapping(), "evolved": evolved.to_mapping()},
        paired_bootstrap(seed_p, evolved_p, labels).to_mapping(),
        pass_verdict(seed, evolved).to_mapping(),
        reference,
    )


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
        backend: ``llama_cpp``, ``vllm`` or ``ollama``.
        model: The served model id.
        pins: Dataset, server and weights pins.
        identity: The experiment identity mapping.

    Returns:
        A JSON-serializable receipt. It keeps both texts verbatim, records
        ``components``, ``seed_parts`` and ``evolved_parts`` from
        ``run.parts`` (``instructions`` only when None, #363) and
        ``run.part_table`` (#369), and pins the mappings by digest in
        ``wording_digests`` (#362). A ``Choice`` run's metrics add ``kappa``
        and its ``reference_133`` is None.

    Raises:
        ValueError: When ``run.split`` is not ``test``, for example a run of
            ``ValidationRows`` (#339), or ``run.parts`` holds other texts.
    """
    if run.split != HELD_OUT_SPLIT:
        msg = f"the #309 receipt needs a {HELD_OUT_SPLIT!r} run, not {run.split!r}"
        raise ValueError(msg)
    complete = run.stopped is None and run.rows > 0
    metrics = bootstrap = verdict = reference = None
    if complete:
        metrics, bootstrap, verdict, reference = _measures(run)
    parts = receipt_parts(seed_text, evolved_text, run.parts)
    return {
        "issue": 309,
        "backend": backend,
        "model": model,
        **wording_fields(parts, run.part_table),
        "rows": run.rows,
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
        ]
        + [p.to_mapping() for p in run.choice_pairs],
        "per_call": [r.to_mapping() for r in run.call_records],
        "call_summary": call_summary(run.call_records),
        "pins": dict(pins),
        "identity": dict(identity),
    }


def evolution_artifact(
    run: WordingRun,
    *,
    config: WordingRunConfig,
    train: Sequence[AnyRow],
    validation: Sequence[AnyRow],
    call_records: Sequence[CallRecord] = (),
) -> dict[str, Any]:
    """Record the seed and evolved wording and how they were chosen.

    Args:
        run: The evolution run.
        config: The run settings; a ``BaseLlm`` reflector is recorded by its
            ``model`` name.
        train: The train records or rows gepa-adk reflected on.
        validation: The validation records or rows gepa-adk selected on.
        call_records: Per-call latency and input tokens of the judge, for
            example ``TimedJudgePort.records`` (#327).

    Returns:
        A JSON-serializable artifact with both ``instructions`` texts, the
        selection, both full part mappings and their digests (#363), the
        seed's ``part_table`` (#369), the
        per-part length caps, the settings, the validation record ids,
        gepa-adk's result and the judge calls.
    """
    reflector = config.reflector
    caps = part_caps(run.seed_parts, run.components, config.length_ratio)
    return {
        "issue": 309,
        **wording_fields(run.parts, run.part_table),
        "reflector": reflector if isinstance(reflector, str) else reflector.model,
        "judge_model": config.judge_model,
        "train_rows": len(train),
        "validation_ids": [as_row(r).record_id for r in validation],
        "length_cap": caps,
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
