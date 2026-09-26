"""Offline TPJEP v0 runner through ``JudgmentPort`` (#106).

Writes [#131](https://github.com/Alberto-Codes/typevet/issues/131) attempt records
via judgment calls. Protocol pin: thinking off, permutations=1, argmax.

Examples:
    ```python
    from tests.fixtures.judgment_contract import ContractJudgmentFake
    from typevet.domain.judgment_answers import NoulAnswer
    from typevet.eval_tpjep_loader import load_eight_task_fixture
    from typevet.eval_tpjep_runner import TpjepRunConfig, run_tpjep_tasks

    tasks = load_eight_task_fixture(fixture_text)
    fake = ContractJudgmentFake(answers={"answer": NoulAnswer(noul=0.9)})
    records = run_tpjep_tasks(fake, tasks[:1], config=TpjepRunConfig(model="fake"))
    assert records[0].protocol == "TPJEP-v0"
    ```

See Also:
    - [typevet.eval_tpjep_records][]: JSONL record shape
    - [typevet.eval_tpjep_loader][]: JevBench row loader
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from typevet.domain.errors import (
    JudgmentError,
    JudgmentValidationError,
    ScoringValidationError,
    TransportError,
)
from typevet.domain.judgment_response import JudgmentResponse
from typevet.eval_tpjep_loader import (
    TPJEP_DATASET_GIT_COMMIT,
    TPJEP_LOCAL_CONCAT_HASH,
    TPJEP_LOCAL_CONCAT_RECIPE,
    TPJEP_MANIFEST_HASH,
    TPJEP_MANIFEST_RECIPE,
    TpjepScheduledTask,
    model_inputs_for_task,
)
from typevet.eval_tpjep_outcome import outcome_from_answer, prob_valid
from typevet.eval_tpjep_records import (
    TPJEP_PROTOCOL_V0,
    TpjepAttemptRecord,
    TpjepOutcome,
    TpjepRunSummary,
    summarize_tpjep_records,
)
from typevet.ports.judgment import JudgmentPort


@dataclass(frozen=True, slots=True)
class TpjepRunConfig:
    """Protocol and pin settings for one TPJEP run.

    Attributes:
        model (str): Model id on each attempt record.
        template_class (str | None): Chat template class when known.
        server_build (str | None): llama.cpp build id when known.
        dataset_git_commit (str): JevBench corpus git commit pin.
        dataset_hash (str): Primary manifest hash on each record row.
        dataset_hash_recipe (str): Recipe for ``dataset_hash``.

    Examples:
        ```python
        TpjepRunConfig(model="gemma-test", template_class="gemma")
        ```
    """

    model: str
    template_class: str | None = None
    server_build: str | None = None
    dataset_git_commit: str = TPJEP_DATASET_GIT_COMMIT
    dataset_hash: str = TPJEP_MANIFEST_HASH
    dataset_hash_recipe: str = TPJEP_MANIFEST_RECIPE


@dataclass(frozen=True, slots=True)
class TpjepRunMetadata:
    """Run receipt metadata including both dataset hash pins (#130).

    Attributes:
        protocol (str): ``TPJEP-v0``.
        thinking (bool): Always false for v0.
        permutations (int): Always one for v0.
        decision_rule (str): ``argmax`` for categorical; Score uses EV.
        manifest_hash (str): Published TypeLLM manifest hash.
        manifest_recipe (str): Manifest hash recipe.
        local_concat_hash (str): Local tier-bytes concat hash.
        local_concat_recipe (str): Local concat recipe.
        model (str): Model id for the run.
        template_class (str | None): Template class when known.
        server_build (str | None): Server build when known.

    Examples:
        ```python
        from typevet.eval_tpjep_runner import run_metadata_from_config

        meta = run_metadata_from_config(TpjepRunConfig(model="m"))
        assert meta.thinking is False
        ```
    """

    protocol: str
    thinking: bool
    permutations: int
    decision_rule: str
    manifest_hash: str
    manifest_recipe: str
    local_concat_hash: str
    local_concat_recipe: str
    model: str
    template_class: str | None
    server_build: str | None


@dataclass(frozen=True, slots=True)
class TpjepRunReceipt:
    """Attempt records, summary, and protocol metadata for one run.

    Attributes:
        records (list[TpjepAttemptRecord]): One row per scheduled task.
        summary (TpjepRunSummary): Aggregate counters from ``records``.
        metadata (TpjepRunMetadata): Protocol pin and both dataset hashes.

    Examples:
        ```python
        receipt = run_tpjep_with_receipt(port, tasks, config=config)
        assert receipt.summary.n_scheduled == len(tasks)
        ```
    """

    records: list[TpjepAttemptRecord]
    summary: TpjepRunSummary
    metadata: TpjepRunMetadata


def _failure_record(
    task: TpjepScheduledTask,
    config: TpjepRunConfig,
    *,
    outcome: TpjepOutcome,
    error_type: str,
    error_message: str,
    duration_ms: int,
) -> TpjepAttemptRecord:
    return TpjepAttemptRecord(
        task_id=task.task_id,
        source_tier=task.source_tier,
        question_type=task.question_type,
        model=config.model,
        dataset_git_commit=config.dataset_git_commit,
        dataset_hash_recipe=config.dataset_hash_recipe,
        dataset_hash=config.dataset_hash,
        protocol=TPJEP_PROTOCOL_V0,
        outcome=outcome,
        predicted=None,
        expected=task.expected,
        probabilities=None,
        prob_valid=False,
        correct=None,
        error_type=error_type,
        error_message=error_message,
        usage=None,
        duration_ms=duration_ms,
        server_build=config.server_build,
        template_class=config.template_class,
    )


def _answered_record(
    task: TpjepScheduledTask,
    config: TpjepRunConfig,
    response: JudgmentResponse,
    *,
    duration_ms: int,
) -> TpjepAttemptRecord:
    answer = response.answers[task.question_name]
    predicted, probs, correct = outcome_from_answer(task, answer)
    valid = prob_valid(probs)
    usage = None
    if (
        response.usage.input_tokens is not None
        or response.usage.output_tokens is not None
    ):
        usage = {
            "prompt_tokens": int(response.usage.input_tokens or 0),
            "completion_tokens": int(response.usage.output_tokens or 0),
        }
    return TpjepAttemptRecord(
        task_id=task.task_id,
        source_tier=task.source_tier,
        question_type=task.question_type,
        model=config.model,
        dataset_git_commit=config.dataset_git_commit,
        dataset_hash_recipe=config.dataset_hash_recipe,
        dataset_hash=config.dataset_hash,
        protocol=TPJEP_PROTOCOL_V0,
        outcome="answered" if valid else "prob_invalid",
        predicted=predicted,
        expected=task.expected,
        probabilities=dict(probs) if probs else None,
        prob_valid=valid,
        correct=correct,
        error_type=None,
        error_message=None,
        usage=usage,
        duration_ms=duration_ms,
        server_build=config.server_build,
        template_class=config.template_class,
    )


def run_tpjep_tasks(
    port: JudgmentPort,
    tasks: Sequence[TpjepScheduledTask],
    *,
    config: TpjepRunConfig,
) -> list[TpjepAttemptRecord]:
    """Run scheduled tasks through ``port`` and return attempt records.

    Args:
        port: Offline fake or live ``ScoringJudgmentAdapter``.
        tasks: Rows from ``load_eight_task_fixture`` or compatible loader.
        config: Model and dataset pin settings.

    Returns:
        One record per task in input order.
    """
    records: list[TpjepAttemptRecord] = []
    for task in tasks:
        state, questions = model_inputs_for_task(task)
        started = time.monotonic()
        try:
            response = port.judge(state, questions, config.model)
        except TransportError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            records.append(
                _failure_record(
                    task,
                    config,
                    outcome="transport_failed",
                    error_type="TransportError",
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )
            )
            continue
        except (JudgmentValidationError, ScoringValidationError) as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            records.append(
                _failure_record(
                    task,
                    config,
                    outcome="schema_invalid",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )
            )
            continue
        except JudgmentError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            records.append(
                _failure_record(
                    task,
                    config,
                    outcome="schema_invalid",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )
            )
            continue
        duration_ms = int((time.monotonic() - started) * 1000)
        records.append(
            _answered_record(task, config, response, duration_ms=duration_ms)
        )
    return records


def run_metadata_from_config(config: TpjepRunConfig) -> TpjepRunMetadata:
    """Build protocol metadata for a run receipt.

    Returns:
        Frozen metadata with both dataset hash pins from #130.
    """
    return TpjepRunMetadata(
        protocol=TPJEP_PROTOCOL_V0,
        thinking=False,
        permutations=1,
        decision_rule="argmax",
        manifest_hash=TPJEP_MANIFEST_HASH,
        manifest_recipe=TPJEP_MANIFEST_RECIPE,
        local_concat_hash=TPJEP_LOCAL_CONCAT_HASH,
        local_concat_recipe=TPJEP_LOCAL_CONCAT_RECIPE,
        model=config.model,
        template_class=config.template_class,
        server_build=config.server_build,
    )


def run_tpjep_with_receipt(
    port: JudgmentPort,
    tasks: Sequence[TpjepScheduledTask],
    *,
    config: TpjepRunConfig,
) -> TpjepRunReceipt:
    """Run tasks and return records, summary, and metadata.

    Returns:
        Receipt with attempt rows, summary, and protocol metadata.
    """
    records = run_tpjep_tasks(port, tasks, config=config)
    summary = summarize_tpjep_records(records)
    return TpjepRunReceipt(
        records=records,
        summary=summary,
        metadata=run_metadata_from_config(config),
    )
