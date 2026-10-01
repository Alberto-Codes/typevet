"""Unit checks for the knobs of the live wording-evolution test (#365).

The live test reads ``TYPEVET_WORDING_COMPONENTS`` and
``TYPEVET_WORDING_SEED_CRITERIA``. These checks import its helpers and drive
them with a scripted port and a scripted reflector, so no call leaves the
process.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_live_knobs.py
    ```

See Also:
    - [typevet_evals.wording.parts][]: the part names and ``artifact_parts``
    - [typevet_evals.wording.comparison][]: ``evolved_parts_for``
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from judgevet import NoulAnswer, SystemOneResponse, Usage

from evals.tests.live import test_wording_evolution_live as live
from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.experiment_identity import PromptSpec
from typevet_evals.wording.comparison import evolved_parts_for, evolved_text_for
from typevet_evals.wording.digests import wording_fields
from typevet_evals.wording.held_out import HeldOutRows, score_held_out
from typevet_evals.wording.parts import WordingParts, receipt_parts
from typevet_evals.wording.runner import WordingRunConfig

pytestmark = pytest.mark.unit

KEY = live.PRIMARY_NOUL_NAME
SEED_TEXT = live.SEED_TEXT
TRUE_TEXT = (
    "The message tries to deceive the reader into money, credentials or an "
    "unsafe action."
)
FALSE_TEXT = "The message is an ordinary personal, commercial or informational text."
SEED_PARTS = {
    "instructions": SEED_TEXT,
    "criteria_true": TRUE_TEXT,
    "criteria_false": FALSE_TEXT,
}
ARM_A = {"TYPEVET_WORDING_COMPONENTS": "criteria_true,criteria_false"}
ARM_A |= {"TYPEVET_WORDING_SEED_CRITERIA": "1"}
EVOLVED_TRUE = "It asks for money or a password."
EVOLVED_FALSE = "It is an everyday message."


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
HELD_OUT = (
    _record("Send gift cards", True, "test"),
    _record("Dinner at six", False, "test"),
)


@dataclass
class ScriptedPort:
    """Answer 0.5 to every question and record each question sent.

    Attributes:
        nouls (list[Any]): Each call's question.
    """

    nouls: list[Any] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the question and answer 0.5.

        Returns:
            One ``Noul`` answer under ``KEY``.
        """
        self.nouls.append(questions[KEY])
        return SystemOneResponse(
            model=model, usage=Usage(), answers={KEY: NoulAnswer(noul=0.5)}
        )


class ScriptedReflector(BaseLlm):
    """A reflection model that always proposes one short text.

    Attributes:
        proposal (str): The proposed text.
    """

    model: str = "scripted-reflector"
    proposal: str = "It is a con."

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Yield the proposal.

        Yields:
            One text response.
        """
        part = types.Part.from_text(text=self.proposal)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _config(tmp_path: Path) -> WordingRunConfig:
    """Return a one-iteration configuration over the scripted reflector."""
    return WordingRunConfig(
        reflector=ScriptedReflector(),
        judge_model="fake-judge",
        max_iterations=1,
        checkpoint_path=tmp_path / "checkpoint.json",
        seed=0,
    )


def _artifact(**changes: Any) -> dict[str, Any]:
    """Return a minimal #363 arm A evolution artifact of the Gemma judge."""
    evolved = SEED_PARTS | {
        "criteria_true": EVOLVED_TRUE,
        "criteria_false": EVOLVED_FALSE,
    }
    artifact: dict[str, Any] = {
        "seed_text": SEED_TEXT,
        "evolved_text": SEED_TEXT,
        "components": ["criteria_true", "criteria_false"],
        "seed_parts": dict(SEED_PARTS),
        "evolved_parts": evolved,
        "judge_provider": "gemma",
        "valid": True,
        "budget_refusals": 0,
    }
    return artifact | changes


def _legacy(evolved_text: str = "Does it ask for money?") -> dict[str, Any]:
    """Return a #252-shaped artifact: no parts fields."""
    return {
        "seed_text": SEED_TEXT,
        "evolved_text": evolved_text,
        "judge_provider": "gemma",
        "valid": True,
        "budget_refusals": 0,
    }


def test_the_seed_criteria_are_the_pre_registered_texts() -> None:
    assert dict(live.SEED_CRITERIA) == {"true": TRUE_TEXT, "false": FALSE_TEXT}


@pytest.mark.parametrize("raw", ["", "   "])
def test_the_components_knob_defaults_to_instructions(raw: str) -> None:
    environ = {} if not raw else {"TYPEVET_WORDING_COMPONENTS": raw}

    assert live.wording_components(environ) == ("instructions",)


def test_the_components_knob_reads_comma_separated_part_names() -> None:
    environ = {"TYPEVET_WORDING_COMPONENTS": " criteria_true , criteria_false "}

    assert live.wording_components(environ) == ("criteria_true", "criteria_false")


@pytest.mark.parametrize(
    "raw",
    ["is_scam_secret_text", "instructions,", "instructions,option_block", ","],
)
def test_the_components_knob_refuses_a_name_outside_the_noul_parts(raw: str) -> None:
    with pytest.raises(ValueError, match="TYPEVET_WORDING_COMPONENTS") as caught:
        live.wording_components({"TYPEVET_WORDING_COMPONENTS": raw})

    assert "is_scam_secret_text" not in str(caught.value)
    assert "option_block" not in str(caught.value)


def test_the_default_seed_is_the_252_seed_without_criteria() -> None:
    seed = live.wording_seed({})

    assert seed.instructions == SEED_TEXT
    assert seed.criteria is None
    zero = live.wording_seed({"TYPEVET_WORDING_SEED_CRITERIA": "0"})
    assert (zero.instructions, zero.criteria) == (SEED_TEXT, None)


def test_the_seed_criteria_knob_adds_the_criteria_texts() -> None:
    seed = live.wording_seed({"TYPEVET_WORDING_SEED_CRITERIA": "1"})

    assert seed.instructions == SEED_TEXT
    assert seed.criteria == {"true": TRUE_TEXT, "false": FALSE_TEXT}


def test_the_seed_criteria_knob_refuses_another_value() -> None:
    with pytest.raises(ValueError, match="TYPEVET_WORDING_SEED_CRITERIA"):
        live.wording_seed({"TYPEVET_WORDING_SEED_CRITERIA": "yes"})


def test_a_criteria_selection_without_seed_criteria_is_refused() -> None:
    environ = {"TYPEVET_WORDING_COMPONENTS": "criteria_true"}

    with pytest.raises(ValueError, match="'criteria_true' is not a part of the seed"):
        live.evolution_inputs(environ)


def test_the_default_evolution_sends_instructions_only(tmp_path: Path) -> None:
    port = ScriptedPort()

    run = asyncio.run(
        live.evolve_from_env(
            port, {}, train=TRAIN, validation=VALIDATION, config=_config(tmp_path)
        )
    )

    assert run.components == ("instructions",)
    assert dict(run.seed_parts) == {"instructions": SEED_TEXT}
    assert port.nouls
    assert all(noul.criteria is None for noul in port.nouls)


def test_arm_a_evolves_the_criteria_and_freezes_the_instructions(
    tmp_path: Path,
) -> None:
    port = ScriptedPort()

    run = asyncio.run(
        live.evolve_from_env(
            port, ARM_A, train=TRAIN, validation=VALIDATION, config=_config(tmp_path)
        )
    )

    assert run.components == ("criteria_true", "criteria_false")
    assert dict(run.seed_parts) == SEED_PARTS
    assert run.evolved_parts["instructions"] == SEED_TEXT
    assert {str(noul.instructions) for noul in port.nouls} == {SEED_TEXT}
    assert {noul.criteria["true"] for noul in port.nouls} >= {TRUE_TEXT}


def test_arm_b_evolves_all_three_parts(tmp_path: Path) -> None:
    environ = {
        "TYPEVET_WORDING_COMPONENTS": "instructions,criteria_true,criteria_false",
        "TYPEVET_WORDING_SEED_CRITERIA": "1",
    }

    run = asyncio.run(
        live.evolve_from_env(
            ScriptedPort(),
            environ,
            train=TRAIN,
            validation=VALIDATION,
            config=_config(tmp_path),
        )
    )

    assert run.components == ("instructions", "criteria_true", "criteria_false")
    assert dict(run.seed_parts) == SEED_PARTS


def test_a_legacy_artifact_gives_the_252_instructions_only_parts() -> None:
    seed, parts = live.held_out_parts(_legacy(), {})

    assert seed.criteria is None
    assert parts == WordingParts.from_texts(SEED_TEXT, "Does it ask for money?")
    assert wording_fields(parts) == wording_fields(
        receipt_parts(SEED_TEXT, "Does it ask for money?", None)
    )


def test_arm_0_carries_the_seed_criteria_into_both_arms() -> None:
    seed, parts = live.held_out_parts(_legacy(), {"TYPEVET_WORDING_SEED_CRITERIA": "1"})

    assert seed.criteria == {"true": TRUE_TEXT, "false": FALSE_TEXT}
    assert parts.components == ("instructions",)
    assert dict(parts.seed) == SEED_PARTS
    assert dict(parts.evolved) == SEED_PARTS | {
        "instructions": "Does it ask for money?"
    }


def test_the_held_out_parts_come_from_the_artifact_evolved_parts() -> None:
    _, parts = live.held_out_parts(_artifact(), ARM_A)

    assert parts.components == ("criteria_true", "criteria_false")
    assert dict(parts.seed) == SEED_PARTS
    assert parts.evolved["criteria_true"] == EVOLVED_TRUE
    assert parts.evolved["criteria_false"] == EVOLVED_FALSE


def test_the_held_out_run_sends_each_arm_its_own_criteria() -> None:
    seed, parts = live.held_out_parts(_artifact(), ARM_A)
    port = ScriptedPort()

    run = score_held_out(
        port,
        seed,
        KEY,
        evolved=parts,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="fake-judge",
        failures=(),
    )

    seed_arm, evolved_arm = port.nouls[0], port.nouls[1]
    assert dict(seed_arm.criteria) == {"true": TRUE_TEXT, "false": FALSE_TEXT}
    assert dict(evolved_arm.criteria) == {
        "true": EVOLVED_TRUE,
        "false": EVOLVED_FALSE,
    }
    assert run.parts == parts


@pytest.mark.parametrize(
    "components", [["is_scam_secret_text"], ["no_thinking_prefill"], "instructions"]
)
def test_an_artifact_part_outside_the_noul_parts_is_refused_value_free(
    components: object,
) -> None:
    with pytest.raises(ValueError, match="components") as caught:
        live.held_out_parts(_artifact(components=components), ARM_A)

    assert "is_scam_secret_text" not in str(caught.value)
    assert "no_thinking_prefill" not in str(caught.value)


def test_an_artifact_from_another_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="seed_parts"):
        live.held_out_parts(_artifact(), {})


def test_an_artifact_whose_frozen_part_changed_is_refused() -> None:
    tampered = _artifact()
    tampered["evolved_parts"] = tampered["evolved_parts"] | {"instructions": "Other"}
    tampered["evolved_text"] = "Other"

    with pytest.raises(ValueError, match="frozen part 'instructions'"):
        live.held_out_parts(tampered, ARM_A)


def test_the_default_prompt_specs_have_no_criteria() -> None:
    parts = WordingParts.from_texts(SEED_TEXT, "Does it ask for money?")

    assert live.prompt_specs(parts) == (
        PromptSpec("seed", ("true", "false"), SEED_TEXT, {}),
        PromptSpec("evolved", ("true", "false"), "Does it ask for money?", {}),
    )


def test_the_prompt_specs_carry_each_arm_criteria() -> None:
    _, parts = live.held_out_parts(_artifact(), ARM_A)

    seed_spec, evolved_spec = live.prompt_specs(parts)

    assert seed_spec.instructions == evolved_spec.instructions == SEED_TEXT
    assert dict(seed_spec.criteria) == {"true": TRUE_TEXT, "false": FALSE_TEXT}
    assert dict(evolved_spec.criteria) == {
        "true": EVOLVED_TRUE,
        "false": EVOLVED_FALSE,
    }


def test_the_comparison_accepts_an_arm_a_artifact_with_frozen_instructions() -> None:
    seed, _ = live.held_out_parts(_artifact(), ARM_A)

    parts = live.comparison_parts(_artifact(), seed, backend="llama_cpp")

    assert parts.evolved["criteria_true"] == EVOLVED_TRUE
    with pytest.raises(ValueError, match="same as the seed"):
        evolved_text_for(_artifact(), backend="llama_cpp", seed_text=SEED_TEXT)


def test_the_comparison_of_a_legacy_artifact_matches_evolved_text_for() -> None:
    seed = live.wording_seed({})

    parts = live.comparison_parts(_legacy(), seed, backend="llama_cpp")

    assert parts.evolved_text == evolved_text_for(
        _legacy(), backend="llama_cpp", seed_text=SEED_TEXT
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"judge_provider": "jev"}, "judge_provider"),
        ({"valid": False}, "not valid"),
        ({"budget_refusals": 2}, "budget_refusals"),
    ],
)
def test_the_comparison_parts_keep_the_artifact_checks(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evolved_parts_for(
            _artifact(**changes), backend="llama_cpp", seed_parts=SEED_PARTS
        )


def test_the_comparison_refuses_parts_equal_to_the_seed() -> None:
    same = _artifact(evolved_parts=dict(SEED_PARTS))

    with pytest.raises(ValueError, match="same as the seed"):
        evolved_parts_for(same, backend="llama_cpp", seed_parts=SEED_PARTS)


def test_an_existing_receipt_path_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "receipt.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TYPEVET_WORDING_HELD_OUT_RECEIPT", str(existing))

    with pytest.raises(pytest.fail.Exception, match="must name a new file"):
        live._required_path("TYPEVET_WORDING_HELD_OUT_RECEIPT", new=True)
