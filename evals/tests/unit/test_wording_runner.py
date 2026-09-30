"""Unit checks for the wording evolution runner (#308).

A fake port answers from the wording and the message, and a fake reflector
proposes scripted wordings, so the gepa-adk engine runs end to end offline.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_runner.py
    ```

See Also:
    - [typevet_evals.wording.runner][]: the runner module
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Noul

from typevet_evals.datasets import difraud
from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording import runner
from typevet_evals.wording.runner import (
    BrierScorer,
    WordingRunConfig,
    brier_score,
    evolve_wording,
    length_cap,
    to_rows,
)

pytestmark = pytest.mark.unit

KEY = "is_scam"
SEED_TEXT = "Is this message a scam?"
BETTER = "Does it ask for money?"
TOO_LONG = SEED_TEXT * 2
SEED = Noul(instructions=SEED_TEXT, criteria=None)


def _record(text: str, scam: bool, split: str) -> DIFrauDRecord:
    """Return one record whose example names ``split``."""
    return DIFrauDRecord(
        record_id(text), replace(map_row(text, int(scam)), split=split)
    )


TRAIN = (_record("Win cash now", True, "train"), _record("See you", False, "train"))
VALIDATION = (
    _record("Claim a prize", True, "validation"),
    _record("Lunch at noon", False, "validation"),
)
HELD_OUT = (_record("Send gift cards", True, "test"),)
SCAMS = {
    r.example.text
    for r in (*TRAIN, *VALIDATION, *HELD_OUT)
    if r.example.label == "scam"
}


@dataclass
class WordingPort:
    """Answer 0.9/0.1 for ``BETTER`` and 0.5 for every other wording.

    Attributes:
        calls (list[tuple[str, str]]): Each call's state and wording.
    """

    calls: list[tuple[str, str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call and answer from the wording and the state.

        Returns:
            One ``Noul`` answer under ``KEY``.
        """
        wording = str(questions[KEY].instructions)
        self.calls.append((state, wording))
        p = (0.9 if state in SCAMS else 0.1) if wording == BETTER else 0.5
        return SystemOneResponse(
            model=model, usage=Usage(), answers={KEY: NoulAnswer(noul=p)}
        )


class ScriptedReflector(BaseLlm):
    """A reflection model that answers the scripted proposals in order.

    Attributes:
        proposals (list[str]): The texts still to propose; the last one repeats.
    """

    model: str = "scripted-reflector"
    proposals: list[str]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Yield the next scripted proposal.

        Yields:
            One text response.
        """
        text = self.proposals.pop(0) if len(self.proposals) > 1 else self.proposals[0]
        part = types.Part.from_text(text=text)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _config(tmp_path: Path, *proposals: str, **kw: Any) -> WordingRunConfig:
    """Return a two-iteration configuration over the scripted reflector."""
    options: dict[str, Any] = {
        "reflector": ScriptedReflector(proposals=list(proposals)),
        "judge_model": "fake-judge",
        "max_iterations": 2,
        "checkpoint_path": tmp_path / "checkpoint.json",
        "seed": 0,
    }
    return WordingRunConfig(**(options | kw))


def _run(port: WordingPort, config: WordingRunConfig, **kw: Any) -> runner.WordingRun:
    """Run the runner over the fixture splits."""
    args: dict[str, Any] = {"train": TRAIN, "validation": VALIDATION} | kw
    return asyncio.run(
        evolve_wording(port=port, seed=SEED, key=KEY, config=config, **args)
    )


@pytest.mark.parametrize(
    ("p", "label", "expected"),
    [(0.8, 1, 0.04), (0.3, 0, 0.09), (0.5, 1, 0.25), (1.0, 0, 1.0), (0.0, 0, 0.0)],
)
def test_brier_score_matches_hand_computed_values(
    p: float, label: int, expected: float
) -> None:
    assert brier_score(p, label) == pytest.approx(expected)


def test_scorer_reads_the_transport_body_against_the_scam_label() -> None:
    scorer = BrierScorer()

    score, meta = scorer.score("Win cash now", json.dumps({"probability": 0.8}), "1")
    async_score, _ = asyncio.run(
        scorer.async_score("See you", json.dumps({"probability": 0.3}), "0")
    )

    assert score == pytest.approx(0.96)
    assert meta == {"brier": pytest.approx(0.04), "probability": 0.8, "label": 1}
    assert async_score == pytest.approx(0.91)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("not json", "1"),
        ('{"p": 0.5}', "1"),
        ('{"probability": 1.5}', "0"),
        ('{"probability": true}', "0"),
        ('{"probability": 0.5}', "scam"),
        ('{"probability": 0.5}', None),
    ],
)
def test_scorer_refuses_a_body_or_label_it_cannot_score(
    output: str, expected: str | None
) -> None:
    with pytest.raises((ValueError, TypeError), match=r"probability|label"):
        BrierScorer().score("x", output, expected)


def test_rows_carry_the_text_and_the_scam_label() -> None:
    assert to_rows(TRAIN) == [
        {"input": "Win cash now", "expected": "1"},
        {"input": "See you", "expected": "0"},
    ]


def test_length_cap_rejects_text_longer_than_one_and_a_half_seeds() -> None:
    check = length_cap("x" * 10)

    assert check(KEY, "y" * 15) is None
    assert check(KEY, "y" * 16) == "proposal has 16 characters; the cap is 15"
    assert check(KEY, "  ") == "empty proposal"


def test_a_too_long_proposal_is_rejected_and_never_sent(tmp_path: Path) -> None:
    port = WordingPort()

    run = _run(port, _config(tmp_path, TOO_LONG, BETTER))

    records = run.result.iteration_history
    assert records[0].skip_reason == "proposal_rejected"
    assert records[0].rejection_reason is not None
    assert "cap is 34" in records[0].rejection_reason
    assert TOO_LONG not in {wording for _, wording in port.calls}
    assert run.evolved_text == BETTER
    assert run.seed_text == SEED_TEXT


def test_selection_scores_the_validation_split(tmp_path: Path) -> None:
    port = WordingPort()

    run = _run(port, _config(tmp_path, BETTER))

    assert run.result.valset_score == pytest.approx(1 - 0.01)
    assert run.result.original_score == pytest.approx(2 * (1 - 0.25))
    assert run.result.final_score == pytest.approx(2 * (1 - 0.01))
    scored = {state for state, wording in port.calls if wording == BETTER}
    assert {r.example.text for r in VALIDATION} <= scored


def test_the_runner_never_sends_a_held_out_text(tmp_path: Path) -> None:
    port = WordingPort()

    _run(port, _config(tmp_path, BETTER))

    sent = {state for state, _ in port.calls}
    assert sent <= {r.example.text for r in (*TRAIN, *VALIDATION)}
    assert not sent & {r.example.text for r in HELD_OUT}


@pytest.mark.parametrize("split", ["train", "validation"])
def test_held_out_records_are_refused_before_any_call(
    tmp_path: Path, split: str
) -> None:
    port = WordingPort()
    splits = {"train": TRAIN, "validation": VALIDATION}
    splits[split] = (*splits[split], *HELD_OUT)

    with pytest.raises(ValueError, match=f"{split} holds a 'test' record"):
        _run(port, _config(tmp_path, BETTER), **splits)
    assert port.calls == []


def test_the_runner_cannot_reach_the_split_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the runner loaded a split")

    for name in (
        "load_splits",
        "build_splits",
        "download_split_jsonl",
        "load_test_split",
    ):
        monkeypatch.setattr(difraud, name, refuse)
    params = set(inspect.signature(evolve_wording).parameters)
    tree = ast.parse(Path(runner.__file__).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }

    _run(WordingPort(), _config(tmp_path, BETTER))

    assert params == {"port", "seed", "key", "train", "validation", "config"}
    assert not names & {"load_splits", "build_splits", "DIFrauDSplits", "difraud"}


def test_configuration_reaches_gepa_adk(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        BETTER,
        reflection_max_trials=1,
        max_concurrent_evals=1,
        max_iterations=1,
    )

    run = _run(WordingPort(), config)

    engine = runner.evolution_config(config, SEED_TEXT)
    assert (engine.reflection_max_trials, engine.max_concurrent_evals) == (1, 1)
    assert engine.proposal_validator is not None
    assert engine.proposal_validator(KEY, TOO_LONG) is not None
    assert run.result.total_iterations == 1
    assert json.loads((tmp_path / "checkpoint.json").read_text())


def test_the_reflection_minibatch_size_reaches_gepa_adk(tmp_path: Path) -> None:
    default = runner.evolution_config(_config(tmp_path, BETTER), SEED_TEXT)
    config = _config(tmp_path, BETTER, reflection_minibatch_size=1, max_iterations=1)

    engine = runner.evolution_config(config, SEED_TEXT)
    run = _run(WordingPort(), config)

    assert default.reflection_minibatch_size is None
    assert engine.reflection_minibatch_size == 1
    assert run.result.total_iterations == 1


def test_a_stop_callback_stops_the_run_with_its_reason(tmp_path: Path) -> None:
    """A caller's stopper, such as a spend-cap check, ends the run (#328)."""
    port = WordingPort()
    seen: list[int] = []

    def after_baseline(state: Any) -> bool:
        seen.append(state.iteration)
        return True

    config = _config(
        tmp_path, BETTER, max_iterations=5, stop_callbacks=(after_baseline,)
    )

    run = _run(port, config)

    assert runner.evolution_config(config, SEED_TEXT).stop_callbacks == [after_baseline]
    assert run.result.stop_reason.value == "stopper_triggered"
    assert run.result.total_iterations == 0
    assert seen == [0]
    assert len(port.calls) == len(TRAIN) + len(VALIDATION), "baseline calls only"


def test_the_default_run_has_no_stop_callback(tmp_path: Path) -> None:
    engine = runner.evolution_config(_config(tmp_path, BETTER), SEED_TEXT)

    assert engine.stop_callbacks == []


def test_the_reflection_prompt_states_the_length_cap(tmp_path: Path) -> None:
    cap = int(1.5 * len(SEED_TEXT))

    engine = runner.evolution_config(_config(tmp_path, BETTER), SEED_TEXT)

    assert engine.reflection_prompt is not None
    assert f"at most {cap} characters" in engine.reflection_prompt
    assert "{component_text}" in engine.reflection_prompt
    assert "{trials}" in engine.reflection_prompt


def test_the_reflector_receives_the_length_cap(tmp_path: Path) -> None:
    seen: list[str] = []

    class RecordingReflector(ScriptedReflector):
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            seen.append(str(llm_request.config.system_instruction))
            async for response in super().generate_content_async(llm_request):
                yield response

    config = replace(
        _config(tmp_path, BETTER, max_iterations=1),
        reflector=RecordingReflector(proposals=[BETTER]),
    )

    _run(WordingPort(), config)

    assert seen
    assert f"at most {int(1.5 * len(SEED_TEXT))} characters" in seen[0]


def test_an_empty_split_or_seed_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation is empty"):
        _run(WordingPort(), _config(tmp_path, BETTER), validation=())
    with pytest.raises(ValueError, match="empty seed"):
        length_cap(" ")


def test_a_generator_split_is_read_once_and_kept(tmp_path: Path) -> None:
    port = WordingPort()

    _run(port, _config(tmp_path, BETTER), train=(r for r in TRAIN))

    assert {r.example.text for r in TRAIN} <= {state for state, _ in port.calls}


def test_resume_without_a_checkpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="resume needs a checkpoint_path"):
        WordingRunConfig(reflector="reflector", judge_model="m", resume=True)


@dataclass
class LoggingHandler:
    """Log each apply and restore of the wrapped gepa-adk component handler.

    Attributes:
        inner (Any): The wrapped handler.
        log (list[tuple[str, str]]): The shared event log.
    """

    inner: Any
    log: list[tuple[str, str]]

    def serialize(self, agent: Any) -> str:
        """Return the wrapped handler's text.

        Returns:
            The current text.
        """
        return self.inner.serialize(agent)

    def apply(self, agent: Any, value: str) -> str:
        """Log and apply ``value``.

        Returns:
            The previous text.
        """
        self.log.append(("apply", value))
        return self.inner.apply(agent, value)

    def restore(self, agent: Any, original: str) -> None:
        """Log and restore ``original``."""
        self.log.append(("restore", original))
        self.inner.restore(agent, original)


@dataclass
class LoggingPort(WordingPort):
    """A ``WordingPort`` that logs each call's wording and holds it briefly.

    Attributes:
        log (list[tuple[str, str]]): The shared event log.
    """

    log: list[tuple[str, str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Log the wording, sleep so calls overlap, then answer.

        Returns:
            The ``WordingPort`` answer.
        """
        self.log.append(("call", str(questions[KEY].instructions)))
        time.sleep(0.01)
        return super().system_one(state, questions, model)


def test_each_evaluation_sees_only_its_own_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple[str, str]] = []
    real = runner.register_mapping_components

    def logging_register(mapping: dict[str, str], *, registry: Any) -> list[str]:
        names = real(mapping, registry=registry)
        for name in names:
            registry.register(name, LoggingHandler(registry.get(name), log))
        return names

    monkeypatch.setattr(runner, "register_mapping_components", logging_register)
    config = _config(tmp_path, "Is it a scam?", BETTER, max_concurrent_evals=5)

    _run(LoggingPort(log=log), config)

    depth, current, seen = 0, "", set()
    for event, text in log:
        if event == "apply":
            assert depth == 0, "a second candidate was applied mid-evaluation"
            depth, current = 1, text
            seen.add(text)
        elif event == "restore":
            depth -= 1
        else:
            assert (depth, text) == (1, current), "a call saw another candidate"
    assert depth == 0
    assert {SEED_TEXT, BETTER} <= seen
