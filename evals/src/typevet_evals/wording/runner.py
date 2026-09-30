"""Evolve question wording with gepa-adk against a Brier score (#308).

``evolve_wording`` puts the seed wording in a mapping, registers the mapping
in a private gepa-adk component registry, and runs ``gepa_adk.evolve`` on a
tool-less agent whose model is the #306 ``WordingTransport``. The reward is
``BrierScorer``: one minus the squared error between the transport's
``{"probability": p}`` and the DIFrauD label (``scam`` is 1). gepa-adk
reflects on the train rows and scores and accepts candidates on the
validation rows. The reflection prompt states the length limit, and a
proposal longer than 1.5 times the seed is rejected before any evaluation.

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
        key="is_scam",
        train=splits.train,
        validation=splits.validation,
        config=WordingRunConfig(reflector="<reflector-model>", judge_model="<judge>"),
    )
    print(run.evolved_text)
    ```

    The real reflector is chosen on #309; ``<reflector-model>`` is a placeholder.

See Also:
    - [typevet_evals.wording.transport][]: the model stand-in the agent uses
    - [typevet_evals.datasets.difraud][]: the train, validation and held-out splits
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa_adk import EvolutionConfig, EvolutionResult, ProposalValidator, evolve
from gepa_adk.adapters.components.component_handlers import ComponentHandlerRegistry
from gepa_adk.adapters.components.mapping_handler import register_mapping_components
from gepa_adk.domain.types import REFLECTION_INSTRUCTION
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm

from typevet_evals.datasets.difraud import DIFrauDRecord
from typevet_evals.wording.transport import JudgePort, SeedNoul, WordingTransport

LENGTH_RATIO = 1.5
"""A proposal may be at most this many times the seed's length."""

POSITIVE_LABEL = "scam"
"""The DIFrauD label whose target probability is 1."""

AGENT_INSTRUCTION = "Answer with the probability."
"""The agent's own instruction; the transport never sends it."""


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


def reflection_prompt(seed_text: str, ratio: float = LENGTH_RATIO) -> str:
    """Return gepa-adk's default reflection prompt with the length limit added.

    Without the limit the reflector proposes texts many times the seed's
    length, and the length cap rejects each one (#309 smoke).

    Args:
        seed_text: The seed wording.
        ratio: The largest allowed proposal length, as a multiple of the seed.

    Returns:
        The default prompt, with its ``{component_text}`` and ``{trials}``
        placeholders, and one line that gives the character limit.
    """
    cap = math.floor(ratio * len(seed_text))
    return (
        f"{REFLECTION_INSTRUCTION}\n"
        f"The improved text must be at most {cap} characters long."
    )


def length_cap(seed_text: str, ratio: float = LENGTH_RATIO) -> ProposalValidator:
    """Return a gepa-adk proposal validator that caps the proposal length.

    Args:
        seed_text: The seed wording.
        ratio: The largest allowed proposal length, as a multiple of the seed.

    Returns:
        A validator that returns a reason for an empty proposal or one longer
        than ``floor(ratio * len(seed_text))`` characters, and None otherwise.

    Raises:
        ValueError: If the seed is empty.
    """
    if not seed_text.strip():
        raise ValueError("empty seed wording")
    cap = math.floor(ratio * len(seed_text))

    def check(component: str, text: str) -> str | None:
        """Return why ``text`` is refused, or None to accept it.

        Args:
            component: The gepa-adk component name; not used.
            text: The proposed wording.

        Returns:
            The reason, or None.
        """
        if not text.strip():
            return "empty proposal"
        if len(text) > cap:
            return f"proposal has {len(text)} characters; the cap is {cap}"
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
        length_ratio (float): The proposal length cap, as a multiple of the seed.

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

    def __post_init__(self) -> None:
        """Refuse ``resume`` without a checkpoint to resume from.

        Raises:
            ValueError: If ``resume`` is true and ``checkpoint_path`` is None.
        """
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume needs a checkpoint_path")


@dataclass(frozen=True, slots=True)
class WordingRun:
    """The seed and evolved wording and gepa-adk's result.

    Attributes:
        seed_text (str): The seed wording.
        evolved_text (str): The wording gepa-adk selected on validation.
        result (EvolutionResult): gepa-adk's result. Its ``valset_score`` is
            the mean of ``1 - brier`` over the validation rows; its
            ``original_score`` and ``final_score`` are sums over them.

    Examples:
        ```python
        run = await evolve_wording(port=port, seed=seed, key="is_scam", ...)
        mapping["is_scam"] = run.evolved_text
        ```
    """

    seed_text: str
    evolved_text: str
    result: EvolutionResult


def evolution_config(config: WordingRunConfig, seed_text: str) -> EvolutionConfig:
    """Return the gepa-adk configuration for one run.

    Args:
        config: The run settings.
        seed_text: The seed wording, which sets the length cap.

    Returns:
        An ``EvolutionConfig`` with the reflector, the reflection minibatch
        size, the length limit in the reflection prompt, the length cap and
        the checkpoint settings.
    """
    return EvolutionConfig(
        max_iterations=config.max_iterations,
        patience=config.patience,
        max_concurrent_evals=config.max_concurrent_evals,
        reflection_model=config.reflector,
        reflection_max_trials=config.reflection_max_trials,
        reflection_minibatch_size=config.reflection_minibatch_size,
        reflection_prompt=reflection_prompt(seed_text, config.length_ratio),
        proposal_validator=length_cap(seed_text, config.length_ratio),
        checkpoint_path=config.checkpoint_path,
        resume=config.resume,
        seed=config.seed,
    )


async def evolve_wording(
    *,
    port: JudgePort,
    seed: SeedNoul,
    key: str,
    train: Iterable[DIFrauDRecord],
    validation: Iterable[DIFrauDRecord],
    config: WordingRunConfig,
) -> WordingRun:
    """Evolve the seed wording on train and select it on validation.

    Args:
        port: The judgevet ``SystemOnePort`` the transport calls.
        seed: The seed judgevet ``Noul``; its instructions are the seed wording.
        key: The question name and the gepa-adk component name.
        train: Records gepa-adk reflects on; each ``split`` is ``train``.
        validation: Records gepa-adk scores and accepts candidates on; each
            ``split`` is ``validation``.
        config: The run settings.

    Returns:
        The seed and evolved wording and gepa-adk's result.

    Raises:
        ValueError: If a split is empty or holds a record of another split.
    """
    trainset, valset = _checked(train, "train"), _checked(validation, "validation")
    seed_text = str(seed.instructions)
    mapping = {key: seed_text}
    registry = ComponentHandlerRegistry()
    components = register_mapping_components(mapping, registry=registry)
    transport = WordingTransport(
        port=port, mapping=mapping, key=key, seed=seed, judge_model=config.judge_model
    )
    agent = LlmAgent(name="wording", model=transport, instruction=AGENT_INSTRUCTION)
    result = await evolve(
        agent,
        trainset,
        valset=valset,
        scorer=BrierScorer(),
        config=evolution_config(config, seed_text),
        components=components,
        registry=registry,
    )
    return WordingRun(seed_text, result.evolved_components[key], result)
