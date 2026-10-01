"""Unit checks for a wording run on a ``Choice`` seed (#369 slice 2).

A scripted port answers ``Choice`` distributions over PubMedQA-shaped
states, and a scripted reflector proposes texts, so gepa-adk runs offline.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_choice_run.py
    ```

See Also:
    - [typevet_evals.wording.scorers][]: the ``Choice`` reward
    - [typevet_evals.wording.rows][]: the seed-agnostic rows
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import LlmRequest
from google.genai import types
from judgevet import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Choice, Noul, Score

from evals.tests.unit.test_wording_runner import ScriptedReflector
from typevet_evals.wording import WordingRunConfig, WordingTransport
from typevet_evals.wording.choice import choice_metrics
from typevet_evals.wording.held_out import (
    HeldOutRows,
    evolution_artifact,
    held_out_receipt,
    score_held_out,
)
from typevet_evals.wording.parts import (
    WordingParts,
    artifact_parts,
    question_mapping,
    seed_mapping,
)
from typevet_evals.wording.rows import WordingRow
from typevet_evals.wording.runner import evolve_wording
from typevet_evals.wording.scorers import ChoiceScorer

pytestmark = pytest.mark.unit

KEY = "answer"
ASK = "Does the abstract answer the question?"
LABELS = ("yes", "no", "maybe")
SEED = Choice(
    instructions=ASK,
    criteria={
        "yes": "The abstract supports the claim.",
        "no": "The abstract refutes the claim.",
        "maybe": "The abstract leaves the claim open.",
    },
)
BETTER_MAYBE = "The results are mixed."
TABLE = {"criteria_yes": "yes", "criteria_no": "no", "criteria_maybe": "maybe"}


def _state(question: str) -> dict[str, Any]:
    """Return a PubMedQA-shaped state."""
    return {"question": question, "contexts": [{"label": "R", "text": "Text."}]}


def _row(question: str, gold: str, split: str) -> WordingRow:
    """Return one row whose gold label is ``gold``."""
    return WordingRow(
        _state(question), gold, record_id=f"pubmedqa:{question}", split=split
    )


def _rows(split: str) -> tuple[WordingRow, ...]:
    """Return one row of each label for ``split``."""
    return tuple(_row(f"{split} {g}", g, split) for g in LABELS)


GOLDS = {f"{s} {g}": g for s in ("train", "validation", "test") for g in LABELS}


def _sharp(gold: str) -> dict[str, float]:
    """Return a distribution with 0.8 on ``gold``."""
    return {label: 0.8 if label == gold else 0.1 for label in LABELS}


FLAT = {"yes": 0.5, "no": 0.3, "maybe": 0.2}


@dataclass
class ChoicePort:
    """Answer sharply when ``maybe`` reads ``BETTER_MAYBE``, else ``FLAT``.

    Attributes:
        calls (list[tuple[Any, Any, dict[str, Any]]]): State, instructions,
            criteria of each call.
    """

    calls: list[tuple[Any, Any, dict[str, Any]]] = field(default_factory=list)

    def system_one(
        self, state: Any, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call and answer one ``Choice`` under ``KEY``.

        Returns:
            One ``ChoiceAnswer``.
        """
        question = questions[KEY]
        self.calls.append((state, question.instructions, dict(question.criteria)))
        gold = GOLDS[state["question"]]
        sharp = question.criteria["maybe"] == BETTER_MAYBE
        probabilities = _sharp(gold) if sharp else dict(FLAT)
        choice = max(probabilities, key=lambda k: probabilities[k])
        answer = ChoiceAnswer(
            choice=choice,
            confidence=probabilities[choice],
            probabilities=probabilities,
        )
        return SystemOneResponse(model=model, usage=Usage(), answers={KEY: answer})


def test_the_choice_scorer_scores_a_scaled_multi_class_brier() -> None:
    scorer = ChoiceScorer(LABELS)
    body = json.dumps({"probabilities": {"yes": 0.7, "no": 0.2, "maybe": 0.1}})

    score, meta = scorer.score("x", body, "yes")
    async_score, _ = asyncio.run(scorer.async_score("x", body, "no"))

    # 1/2 * (0.3^2 + 0.2^2 + 0.1^2) = 0.07
    assert score == pytest.approx(0.93)
    assert meta["brier"] == pytest.approx(0.07)
    assert meta["gold"] == "yes"
    # 1/2 * (0.7^2 + 0.8^2 + 0.1^2) = 0.57
    assert async_score == pytest.approx(0.43)


@pytest.mark.parametrize(
    ("probabilities", "gold"),
    [
        ({"yes": 0.7, "no": 0.3}, "maybe"),
        ({"yes": 0.7, "no": 0.2, "maybe": 0.1}, "other"),
        ({"yes": 0.7, "no": 0.2, "maybe": 0.1, "other": 0.0}, "yes"),
        ({"yes": 1.5, "no": 0.2, "maybe": 0.1}, "yes"),
    ],
    ids=["missing-label", "unknown-gold", "extra-label", "out-of-range"],
)
def test_the_choice_scorer_refuses_a_missing_or_odd_label(
    probabilities: dict[str, float], gold: str
) -> None:
    body = json.dumps({"probabilities": probabilities})

    with pytest.raises(ValueError, match=r"label|probabilit"):
        ChoiceScorer(LABELS).score("x", body, gold)


def test_the_choice_scorer_refuses_a_body_without_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        ChoiceScorer(LABELS).score("x", json.dumps({"probability": 0.5}), "yes")


def _request(text: str) -> LlmRequest:
    """Return a request whose last user text is ``text``."""
    part = types.Part.from_text(text=text)
    return LlmRequest(contents=[types.Content(role="user", parts=[part])])


async def _body(transport: WordingTransport, text: str) -> Any:
    """Return the JSON body of one transport turn."""
    [response] = [r async for r in transport.generate_content_async(_request(text))]
    assert response.content is not None and response.content.parts
    return json.loads(response.content.parts[0].text or "")


def test_the_transport_returns_probabilities_for_a_choice_seed() -> None:
    port = ChoicePort()
    state = _state("train no")
    key = json.dumps(state, sort_keys=True, ensure_ascii=False)
    transport = WordingTransport(
        port=port,
        mapping=question_mapping(SEED),
        question_name=KEY,
        seed=SEED,
        judge_model="m",
        states={key: state},
    )

    body = asyncio.run(_body(transport, key))

    assert body == {"probabilities": FLAT}
    assert port.calls[0][0] == state


@dataclass
class NoulPort:
    """Answer 0.7 to a ``Noul`` under ``is_scam``."""

    def system_one(
        self, state: Any, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Answer one ``Noul``.

        Returns:
            One ``NoulAnswer``.
        """
        answers = {"is_scam": NoulAnswer(noul=0.7)}
        return SystemOneResponse(model=model, usage=Usage(), answers=answers)


def test_the_transport_still_returns_probability_for_a_noul_seed() -> None:
    seed = Noul(instructions="Is it a scam?")
    transport = WordingTransport(
        port=NoulPort(),
        mapping=seed_mapping(seed),
        question_name="is_scam",
        seed=seed,
        judge_model="m",
    )

    assert asyncio.run(_body(transport, "Win cash")) == {"probability": 0.7}


def _config(tmp_path: Path, *proposals: str) -> WordingRunConfig:
    """Return a two-iteration configuration over the scripted reflector."""
    return WordingRunConfig(
        reflector=ScriptedReflector(proposals=list(proposals)),
        judge_model="fake-judge",
        max_iterations=2,
        checkpoint_path=tmp_path / "checkpoint.json",
        seed=0,
    )


def _evolve(port: ChoicePort, tmp_path: Path) -> Any:
    """Evolve ``criteria_maybe`` only over the fixture rows."""
    return asyncio.run(
        evolve_wording(
            port=port,
            seed=SEED,
            question_name=KEY,
            train=_rows("train"),
            validation=_rows("validation"),
            config=_config(tmp_path, BETTER_MAYBE),
            components=("criteria_maybe",),
        )
    )


def test_a_choice_run_evolves_criteria_maybe_and_freezes_the_rest(
    tmp_path: Path,
) -> None:
    port = ChoicePort()

    run = _evolve(port, tmp_path)

    seed_parts = question_mapping(SEED)
    assert run.components == ("criteria_maybe",)
    assert run.evolved_parts == seed_parts | {"criteria_maybe": BETTER_MAYBE}
    # Each sharp row scores 1 - 1/2 * (0.2^2 + 0.1^2 + 0.1^2) = 0.97.
    assert run.result.valset_score == pytest.approx(0.97)
    assert dict(run.part_table) == TABLE
    assert {state["question"] for state, _, _ in port.calls} <= set(GOLDS)
    assert not any(state["question"].startswith("test") for state, _, _ in port.calls)
    for _, instructions, criteria in port.calls:
        assert instructions == SEED.instructions
        assert criteria["yes"] == SEED.criteria["yes"]
        assert criteria["no"] == SEED.criteria["no"]
    artifact = evolution_artifact(
        run,
        config=_config(tmp_path, BETTER_MAYBE),
        train=_rows("train"),
        validation=_rows("validation"),
    )
    assert artifact["part_table"] == TABLE
    assert artifact["validation_ids"] == [r.record_id for r in _rows("validation")]


def test_a_choice_run_refuses_a_gold_label_outside_the_seed(tmp_path: Path) -> None:
    train = (*_rows("train"), _row("train odd", "other", "train"))

    with pytest.raises(ValueError, match="gold"):
        asyncio.run(
            evolve_wording(
                port=ChoicePort(),
                seed=SEED,
                question_name=KEY,
                train=train,
                validation=_rows("validation"),
                config=_config(tmp_path, BETTER_MAYBE),
            )
        )


def _held_out() -> Any:
    """Score the seed and the evolved ``maybe`` part on the test rows."""
    parts = WordingParts(
        ("criteria_maybe",),
        question_mapping(SEED),
        question_mapping(SEED) | {"criteria_maybe": BETTER_MAYBE},
    )
    return score_held_out(
        ChoicePort(),
        SEED,
        KEY,
        evolved=parts,
        rows=HeldOutRows(_rows("test"), frozenset()),
        judge_model="m",
        failures=(RuntimeError,),
    )


def test_a_choice_held_out_receipt_reports_accuracy_brier_ece_and_kappa() -> None:
    run = _held_out()

    receipt = held_out_receipt(
        run,
        seed_text=ASK,
        evolved_text=ASK,
        backend="vllm",
        model="m",
        pins={},
        identity={},
    )

    seed, evolved = receipt["metrics"]["seed"], receipt["metrics"]["evolved"]
    # The flat answer reads yes on every row: one row right of three.
    assert seed["accuracy"] == pytest.approx(1 / 3)
    # (0.19 + 0.39 + 0.49) / 3; each row is 1/2 the squared error sum.
    assert seed["brier"] == pytest.approx(1.07 / 3)
    assert seed["ece"] == pytest.approx(0.5 - 1 / 3)
    assert seed["kappa"] == pytest.approx(0.0)
    assert evolved["accuracy"] == pytest.approx(1.0)
    assert evolved["brier"] == pytest.approx(0.03)
    assert evolved["ece"] == pytest.approx(0.2)
    assert evolved["kappa"] == pytest.approx(1.0)
    assert receipt["rows"] == 3
    assert receipt["part_table"] == TABLE
    assert receipt["reference_133"] is None
    assert set(receipt["bootstrap"]) >= {"ece_difference", "brier_difference"}
    assert receipt["verdict"]["accuracy_ok"] is True
    assert receipt["pairs"][0] == {
        "record_id": "pubmedqa:test yes",
        "gold": "yes",
        "seed": FLAT,
        "evolved": _sharp("yes"),
    }


def test_a_choice_held_out_run_refuses_a_gold_label_outside_the_seed() -> None:
    rows = HeldOutRows((_row("test odd", "other", "test"),), frozenset())

    with pytest.raises(ValueError, match="gold"):
        score_held_out(
            ChoicePort(),
            SEED,
            KEY,
            evolved=WordingParts(
                ("instructions",), question_mapping(SEED), question_mapping(SEED)
            ),
            rows=rows,
            judge_model="m",
            failures=(RuntimeError,),
        )


SCORE = Score(instructions="Rate it.", criteria=["Poor", "Fair", "Good"])


def test_a_score_seed_is_still_refused_by_both_runs(tmp_path: Path) -> None:
    mapping = question_mapping(SCORE)

    with pytest.raises(ValueError, match="Score"):
        seed_mapping(SCORE)
    with pytest.raises(ValueError, match="Score"):
        asyncio.run(
            evolve_wording(
                port=ChoicePort(),
                seed=SCORE,
                question_name=KEY,
                train=_rows("train"),
                validation=_rows("validation"),
                config=_config(tmp_path, "x"),
            )
        )
    with pytest.raises(ValueError, match="Score"):
        score_held_out(
            ChoicePort(),
            SCORE,
            KEY,
            evolved=WordingParts(("instructions",), mapping, mapping),
            rows=HeldOutRows(_rows("test"), frozenset()),
            judge_model="m",
            failures=(RuntimeError,),
        )


def test_artifact_parts_round_trip_a_choice_mapping() -> None:
    seed_parts = question_mapping(SEED)
    evolved = seed_parts | {"criteria_maybe": BETTER_MAYBE}
    parts = WordingParts(("criteria_maybe",), seed_parts, evolved)
    artifact = {
        "seed_text": parts.seed_text,
        "evolved_text": parts.evolved_text,
        "components": list(parts.components),
        "seed_parts": dict(parts.seed),
        "evolved_parts": dict(parts.evolved),
    }

    assert artifact_parts(artifact, seed_parts) == parts
    with pytest.raises(ValueError, match="components"):
        artifact_parts(artifact | {"components": ["criteria_true"]}, seed_parts)


def test_choice_metrics_refuse_bad_rows_and_leave_kappa_undefined() -> None:
    with pytest.raises(ValueError, match="no rows"):
        choice_metrics([], [], LABELS)
    with pytest.raises(ValueError, match="gold label"):
        choice_metrics([FLAT], ["other"], LABELS)
    with pytest.raises(TypeError, match="mapping"):
        ChoiceScorer(LABELS).score("x", json.dumps({"probabilities": [0.5]}), "yes")

    # Every row reads yes and is gold yes, so chance agreement is 1.
    metrics = choice_metrics([FLAT, FLAT], ["yes", "yes"], LABELS)

    assert metrics.accuracy == 1.0
    assert metrics.kappa is None
