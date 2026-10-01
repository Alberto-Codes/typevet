"""Evolve question wording with gepa-adk against a Brier score (#308).

``evolve_wording`` puts every part of the seed ``Noul`` in a mapping
(``instructions``, and ``criteria_true`` and ``criteria_false`` when the
seed has criteria), registers the mapping in a private gepa-adk component
registry, and runs ``gepa_adk.evolve`` on a tool-less agent whose model is
the #306 ``WordingTransport``. ``components`` selects the parts that evolve
(#363); the other parts stay frozen, and the run fails when a frozen part
changed. The reward is ``BrierScorer``: one minus the squared error between
the transport's ``{"probability": p}`` and the DIFrauD label (``scam`` is
1). gepa-adk reflects on the train rows and scores and accepts candidates
on the validation rows. The reflection prompt names each evolvable part,
its role from ``part_roles(seed)`` and its length limit, and a proposal longer than 1.5 times its
seed part is rejected before any evaluation.
Caller stoppers, such as a judge spend-cap check (#328), reach gepa-adk
through ``WordingRunConfig.stop_callbacks``.

The runner takes train and validation records only. It never imports a split
loader, and it refuses a record whose ``split`` is not the one it expects, so
a held-out (``test``) record cannot reach the port.

The transport reads the shared mapping at call time, so the runner depends
on one gepa-adk 2.6.0 property: two different candidates are never
evaluated at the same time. ``ADKAdapter.evaluate`` applies one candidate,
gathers its rows and restores the mapping in ``finally``; every engine call
to ``adapter.evaluate`` is awaited in turn, and the engine and merge
proposer start no concurrent tasks. ``max_concurrent_evals`` therefore
overlaps rows of one candidate only. A unit test fails if two candidates
interleave.

Examples:
    ```python
    from typevet_evals.datasets.difraud import load_splits
    from typevet_evals.wording.runner import WordingRunConfig, evolve_wording

    splits = load_splits(seed=0)
    run = await evolve_wording(
        port=port,
        seed=seed_noul,
        question_name="is_scam",
        train=splits.train,
        validation=splits.validation,
        config=WordingRunConfig(reflector="<reflector-model>", judge_model="<judge>"),
        components=("instructions", "criteria_true"),
    )
    print(run.evolved_parts)
    ```

    The real reflector is chosen on #309; ``<reflector-model>`` is a placeholder.

See Also:
    - [typevet_evals.wording.transport][]: the model stand-in the agent uses
    - [typevet_evals.wording.parts][]: the part names and the frozen-part check
    - [typevet_evals.datasets.difraud][]: the train, validation and held-out splits
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa_adk import EvolutionConfig, EvolutionResult, ProposalValidator, evolve
from gepa_adk.adapters.components.component_handlers import ComponentHandlerRegistry
from gepa_adk.adapters.components.mapping_handler import register_mapping_components
from gepa_adk.domain.types import REFLECTION_INSTRUCTION
from gepa_adk.ports.stopper import StopperProtocol
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm

from typevet_evals.datasets.difraud import DIFrauDRecord
from typevet_evals.wording.parts import (
    INSTRUCTIONS,
    PART_ROLES,
    SeedNoul,
    WordingParts,
    check_selection,
    part_roles,
    seed_mapping,
)
from typevet_evals.wording.transport import JudgePort, WordingTransport

LENGTH_RATIO = 1.5
"""A proposal may be at most this many times the length of its seed part."""

POSITIVE_LABEL = "scam"
"""The DIFrauD label whose target probability is 1."""

AGENT_INSTRUCTION = "Answer with the probability."
"""The agent's own instruction; the transport never sends it."""

_PARTS_INTRO = (
    "The component text is one part of a yes/no question that a judge answers"
    " about a message. The parts that evolve are:"
)
_PARTS_RULE = (
    "The component text is one of these parts. Keep its role, and keep it"
    " within the limit of that part."
)


def brier_score(probability: float, label: int) -> float:
    """Return the Brier score of one probability against a 0 or 1 label.

    Args:
        probability: The predicted probability of the positive label.
        label: 1 for the positive label, 0 otherwise.

    Returns:
        ``(probability - label) ** 2``; 0 is a perfect answer, 1 the worst.
    """
    return (probability - label) ** 2


def _probability(output: str) -> float:
    """Read ``p`` from the transport body ``{"probability": p}``.

    Args:
        output: The agent's final text.

    Returns:
        The probability, between 0 and 1.

    Raises:
        ValueError: If the text is not that body or ``p`` is out of range.
        TypeError: If ``p`` is not a number.
    """
    try:
        value = json.loads(output)["probability"]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"no probability in {output!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"probability {value!r} is not a number")
    if not 0.0 <= value <= 1.0 or math.isnan(value):
        raise ValueError(f"probability {value!r} is not between 0 and 1")
    return float(value)


class BrierScorer:
    """A gepa-adk ``Scorer``: one minus the Brier score, so higher is better.

    ``expected`` is the row's label, ``"1"`` for scam and ``"0"`` for legit.
    A body or label it cannot read raises ``ValueError``; gepa-adk then counts
    the row as a failed evaluation that scores 0.

    Examples:
        ```python
        score, meta = BrierScorer().score("Win cash", '{"probability": 0.8}', "1")
        # score == 0.96, meta["brier"] == 0.04
        ```
    """

    def score(
        self, input_text: str, output: str, expected: str | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Score one transport body against its label.

        Args:
            input_text: The message text; not used.
            output: The transport body ``{"probability": p}``.
            expected: ``"1"`` or ``"0"``.

        Returns:
            ``1 - brier`` and the Brier score, probability and label.

        Raises:
            ValueError: If the label is not ``"1"`` or ``"0"``.
        """
        if expected not in ("0", "1"):
            raise ValueError(f"label {expected!r} is not '0' or '1'")
        label = int(expected)
        probability = _probability(output)
        brier = brier_score(probability, label)
        return 1.0 - brier, {"brier": brier, "probability": probability, "label": label}

    async def async_score(
        self, input_text: str, output: str, expected: str | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Score one transport body against its label.

        Args:
            input_text: The message text; not used.
            output: The transport body ``{"probability": p}``.
            expected: ``"1"`` or ``"0"``.

        Returns:
            The same result as ``score``.
        """
        return self.score(input_text, output, expected)


def part_caps(
    seed_parts: Mapping[str, str],
    components: Iterable[str] | None = None,
    ratio: float = LENGTH_RATIO,
) -> dict[str, int]:
    """Return the length cap of each evolvable part.

    Args:
        seed_parts: The seed's full part mapping.
        components: The evolvable part names; None selects every part.
        ratio: The largest allowed proposal length, as a multiple of the
            seed part.

    Returns:
        Part name to ``floor(ratio * len(seed part))``, in selection order.

    Raises:
        ValueError: If the selection is not valid or a selected seed part is
            empty; the message names the part only.
    """
    selection = check_selection(
        seed_parts if components is None else components, seed_parts
    )
    for name in selection:
        if not seed_parts[name].strip():
            msg = f"empty seed part {name!r}"
            raise ValueError(msg)
    return {name: math.floor(ratio * len(seed_parts[name])) for name in selection}


def reflection_prompt(
    seed_parts: Mapping[str, str],
    components: Iterable[str] | None = None,
    ratio: float = LENGTH_RATIO,
    *,
    roles: Mapping[str, str] = PART_ROLES,
) -> str:
    """Return gepa-adk's default reflection prompt with the parts added.

    gepa-adk takes one prompt per run, so the prompt lists every evolvable
    part with its role and length limit. Without a limit the reflector
    proposes texts many times the seed's length, and the length cap rejects
    each one (#309 smoke).

    Args:
        seed_parts: The seed's full part mapping.
        components: The evolvable part names; None selects every part.
        ratio: The largest allowed proposal length, as a multiple of the
            seed part.
        roles: The role of each part, as ``part_roles(seed)`` gives; the
            ``Noul`` roles by default.

    Returns:
        The default prompt, with its ``{component_text}`` and ``{trials}``
        placeholders, and one line per evolvable part.
    """
    caps = part_caps(seed_parts, components, ratio)
    lines = [
        f"- {name} ({roles[name]}): at most {cap} characters."
        for name, cap in caps.items()
    ]
    return "\n".join([REFLECTION_INSTRUCTION, _PARTS_INTRO, *lines, _PARTS_RULE])


def length_cap(
    seed_parts: Mapping[str, str],
    components: Iterable[str] | None = None,
    ratio: float = LENGTH_RATIO,
) -> ProposalValidator:
    """Return a gepa-adk proposal validator that caps each part by its seed.

    Args:
        seed_parts: The seed's full part mapping.
        components: The evolvable part names; None selects every part.
        ratio: The largest allowed proposal length, as a multiple of the
            seed part.

    Returns:
        A validator that returns a reason for a part that is not evolvable,
        an empty proposal or one longer than its part's cap, and None
        otherwise.
    """
    caps = part_caps(seed_parts, components, ratio)

    def check(component: str, text: str) -> str | None:
        """Return why ``text`` is refused, or None to accept it.

        Args:
            component: The gepa-adk component name, a part name.
            text: The proposed text.

        Returns:
            The reason, or None.
        """
        if component not in caps:
            return f"{component!r} is not an evolvable part"
        if not text.strip():
            return "empty proposal"
        if len(text) > caps[component]:
            return f"proposal has {len(text)} characters; the cap is {caps[component]}"
        return None

    return check


def to_rows(records: Sequence[DIFrauDRecord]) -> list[dict[str, Any]]:
    """Return gepa-adk rows: the message text and the scam label as ``"1"``/``"0"``.

    Args:
        records: DIFrauD records.

    Returns:
        One ``{"input": text, "expected": label}`` row per record.
    """
    return [
        {
            "input": r.example.text,
            "expected": "1" if r.example.label == POSITIVE_LABEL else "0",
        }
        for r in records
    ]


def _checked(records: Iterable[DIFrauDRecord], split: str) -> list[dict[str, Any]]:
    """Return the rows of ``records`` after checking each came from ``split``.

    Args:
        records: The records given for ``split``.
        split: ``train`` or ``validation``.

    Returns:
        The gepa-adk rows.

    Raises:
        ValueError: If ``records`` is empty or holds a record of another split.
    """
    records = list(records)
    if not records:
        raise ValueError(f"{split} is empty")
    for record in records:
        if record.example.split != split:
            raise ValueError(f"{split} holds a {record.example.split!r} record")
    return to_rows(records)


@dataclass(frozen=True, slots=True)
class WordingRunConfig:
    """Settings for one wording evolution.

    Attributes:
        reflector (str | BaseLlm): The gepa-adk reflection model: a model string,
            or a ``BaseLlm`` such as a test fake.
        judge_model (str): The model name the transport sends to the port.
        max_iterations (int): gepa-adk iterations after the baseline.
        patience (int): Iterations without improvement before an early stop.
        reflection_max_trials (int | None): The trials each reflection call sees.
        reflection_minibatch_size (int | None): Train rows each proposal is
            first scored on; None scores the full train split each iteration.
        max_concurrent_evals (int): Evaluations gepa-adk runs at the same time.
        checkpoint_path (Path | None): The JSON file gepa-adk checkpoints to.
        resume (bool): Continue from ``checkpoint_path``.
        seed (int | None): The seed of gepa-adk's engine decisions.
        length_ratio (float): The proposal length cap, as a multiple of the
            seed part.
        stop_callbacks (tuple[StopperProtocol, ...]): gepa-adk stoppers, for
            example a judge spend-cap check (#328). gepa-adk checks them after
            the baseline and after each iteration, and a stop reports
            ``stopper_triggered``.

    Examples:
        ```python
        config = WordingRunConfig(reflector="<reflector-model>", judge_model="<judge>")
        ```
    """

    reflector: str | BaseLlm
    judge_model: str
    max_iterations: int = 10
    patience: int = 5
    reflection_max_trials: int | None = 8
    reflection_minibatch_size: int | None = None
    max_concurrent_evals: int = 5
    checkpoint_path: Path | None = None
    resume: bool = False
    seed: int | None = 0
    length_ratio: float = LENGTH_RATIO
    stop_callbacks: tuple[StopperProtocol, ...] = ()

    def __post_init__(self) -> None:
        """Refuse ``resume`` without a checkpoint to resume from.

        Raises:
            ValueError: If ``resume`` is true and ``checkpoint_path`` is None.
        """
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume needs a checkpoint_path")


@dataclass(frozen=True, slots=True)
class WordingRun:
    """The seed and evolved parts and gepa-adk's result.

    Attributes:
        parts (WordingParts): The evolved selection and the seed and evolved
            full mappings. The evolved selected parts are gepa-adk's
            ``evolved_components``, unchanged.
        result (EvolutionResult): gepa-adk's result. Its ``valset_score`` is
            the mean of ``1 - brier`` over the validation rows; its
            ``original_score`` and ``final_score`` are sums over them.

    Examples:
        ```python
        run = await evolve_wording(port=port, seed=seed, question_name="is_scam", ...)
        mapping.update(run.evolved_parts)
        ```
    """

    parts: WordingParts
    result: EvolutionResult

    @property
    def components(self) -> tuple[str, ...]:
        """Return the evolved selection.

        Returns:
            The part names the run evolved.
        """
        return self.parts.components

    @property
    def seed_parts(self) -> Mapping[str, str]:
        """Return the seed's full mapping.

        Returns:
            Part name to seed text.
        """
        return self.parts.seed

    @property
    def evolved_parts(self) -> Mapping[str, str]:
        """Return the evolved full mapping.

        Returns:
            Part name to evolved text; a frozen part keeps its seed text.
        """
        return self.parts.evolved

    @property
    def seed_text(self) -> str:
        """Return the seed ``instructions`` text.

        Returns:
            The seed wording.
        """
        return self.parts.seed_text

    @property
    def evolved_text(self) -> str:
        """Return the evolved ``instructions`` text.

        Returns:
            The wording gepa-adk selected on validation.
        """
        return self.parts.evolved_text


def evolution_config(
    config: WordingRunConfig,
    seed_parts: Mapping[str, str],
    components: Iterable[str],
    *,
    roles: Mapping[str, str] = PART_ROLES,
) -> EvolutionConfig:
    """Return the gepa-adk configuration for one run.

    Args:
        config: The run settings.
        seed_parts: The seed's full part mapping, which sets the length caps.
        components: The evolvable part names.
        roles: The role of each part, as ``part_roles(seed)`` gives; the
            ``Noul`` roles by default.

    Returns:
        An ``EvolutionConfig`` with the reflector, the reflection minibatch
        size, the parts and limits in the reflection prompt, the per-part
        length cap, the stoppers and the checkpoint settings.
    """
    selection = tuple(components)
    return EvolutionConfig(
        max_iterations=config.max_iterations,
        patience=config.patience,
        max_concurrent_evals=config.max_concurrent_evals,
        reflection_model=config.reflector,
        reflection_max_trials=config.reflection_max_trials,
        reflection_minibatch_size=config.reflection_minibatch_size,
        reflection_prompt=reflection_prompt(
            seed_parts, selection, config.length_ratio, roles=roles
        ),
        proposal_validator=length_cap(seed_parts, selection, config.length_ratio),
        checkpoint_path=config.checkpoint_path,
        resume=config.resume,
        seed=config.seed,
        stop_callbacks=list(config.stop_callbacks),
    )


async def evolve_wording(
    *,
    port: JudgePort,
    seed: SeedNoul,
    question_name: str,
    train: Iterable[DIFrauDRecord],
    validation: Iterable[DIFrauDRecord],
    config: WordingRunConfig,
    components: Sequence[str] = (INSTRUCTIONS,),
) -> WordingRun:
    """Evolve the selected parts on train and select them on validation.

    Args:
        port: The judgevet ``SystemOnePort`` the transport calls.
        seed: The seed judgevet ``Noul``; ``seed_mapping(seed)`` gives its parts,
            and ``part_roles(seed)`` gives their roles in the reflection prompt.
        question_name: The question name sent to the port.
        train: Records gepa-adk reflects on; each ``split`` is ``train``.
        validation: Records gepa-adk scores and accepts candidates on; each
            ``split`` is ``validation``.
        config: The run settings.
        components: The part names gepa-adk evolves; ``instructions`` by
            default. Every other part is frozen.

    Returns:
        The selection, the seed and evolved full mappings and gepa-adk's
        result.

    Raises:
        ValueError: If a split is empty or holds a record of another split,
            the seed is not a ``Noul``, the selection is not valid, or a
            frozen part changed during the run.
    """
    trainset, valset = _checked(train, "train"), _checked(validation, "validation")
    seed_parts = seed_mapping(seed)
    selection = check_selection(components, seed_parts)
    mapping = dict(seed_parts)
    registry = ComponentHandlerRegistry()
    register_mapping_components(mapping, registry=registry)
    transport = WordingTransport(
        port=port,
        mapping=mapping,
        question_name=question_name,
        seed=seed,
        judge_model=config.judge_model,
    )
    agent = LlmAgent(name="wording", model=transport, instruction=AGENT_INSTRUCTION)
    result = await evolve(
        agent,
        trainset,
        valset=valset,
        scorer=BrierScorer(),
        config=evolution_config(config, seed_parts, selection, roles=part_roles(seed)),
        components=list(selection),
        registry=registry,
    )
    evolved = mapping | result.evolved_components
    return WordingRun(WordingParts(selection, seed_parts, evolved), result)
