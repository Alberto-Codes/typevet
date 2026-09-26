"""Questions, states and image controls for the PSAI vision smoke ([#154][i154]).

Two legs share the fixture set.

The **annotation leg** asks ``category`` and ``requires_login`` over the PSAI
``task_name`` state, exactly as the metadata loader does. Hub annotations back
both answers.

The **visual leg** asks the manual ``shows_fox_news_chrome`` Noul three ways for
every row: image present, image omitted, and image swapped with the opposite
visual family. It must not reuse ``task_name``. Every PSAI task text pins its
site (``Only use http://foxnews.com``), which hands the model the answer as
text and would void the controls. ``VISUAL_CONTROL_STATE`` is family-neutral
instead, so pixels are the only family signal in the prompt.

An omitted image is **never** a semantic hit. With no image the model still
answers from its prior, so a match there measures the prior, not the pixels.

Two rules read the visual leg, and they differ in what they assume.
``semantic_hit`` reads one probability against ``NOUL_THRESHOLD``, which only
means something when the Noul is calibrated. ``paired_image_ordering`` compares
the two imaged controls of one row against each other. Their text is identical,
so the difference is the pixels, and no calibration assumption is needed. Prefer
the paired rule as the gate and keep the threshold reading as a diagnostic.

[i154]: https://github.com/Alberto-Codes/typevet/issues/154

Examples:
    Build the control matrix for the vendored rows:

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.psai_vision import load_vision_smoke
    from typevet.evaluation.datasets.psai_vision_controls import control_matrix

    root = Path("tests/fixtures/psai/vision_smoke")
    fixture = load_vision_smoke((root / "manifest.json").read_text())
    controls = control_matrix(fixture.examples)
    assert len(controls) == len(fixture.examples) * 3
    ```

See Also:
    - [typevet.evaluation.datasets.psai_vision][]: fixture loading and provenance
    - [typevet.ports.judgment][]: the ``judge`` surface these questions feed
    - docs/how-to/run-the-psai-vision-smoke.md: the recipe

Attributes:
    VISUAL_CONTROL_STATE (str): Family-neutral state for the visual leg.
    CONDITION_PRESENT (str): Correct screenshot attached.
    CONDITION_OMITTED (str): Same text, no image.
    CONDITION_SWAPPED (str): Same text, opposite-family screenshot.
    CONDITIONS (tuple[str, ...]): The three conditions in run order.
    FAMILY_LEAK_TERMS (tuple[str, ...]): Terms a visual state must never hold.
    NOUL_THRESHOLD (float): Probability at or above which a Noul reads true.
    PAIRED_MARGIN_FLOOR (float): Smallest present-against-swapped margin that
        still counts as image-conditioned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from typevet.domain.judgment_questions import Choice, Noul, Question
from typevet.evaluation.datasets.psai import psai_state, validate_task_state
from typevet.evaluation.datasets.psai_schema import (
    CATEGORY_LABELS,
    METADATA_DECISIONS_SCHEMA,
    PRIMARY_NOUL_NAME,
)
from typevet.evaluation.datasets.psai_vision import (
    ANNOTATION_PROVENANCE,
    FOX_FAMILY,
    MANUAL_VISUAL_PROVENANCE,
    NON_FOX_FAMILY,
    VISUAL_FAMILIES,
    VISUAL_QUESTION_NAME,
    PsaiVisionExample,
)

VISUAL_CONTROL_STATE: Final[str] = (
    "A screen capture taken during a recorded desktop browsing session."
)
CONDITION_PRESENT: Final[str] = "present"
CONDITION_OMITTED: Final[str] = "omitted"
CONDITION_SWAPPED: Final[str] = "swapped"
CONDITIONS: Final[tuple[str, ...]] = (
    CONDITION_PRESENT,
    CONDITION_OMITTED,
    CONDITION_SWAPPED,
)
FAMILY_LEAK_TERMS: Final[tuple[str, ...]] = (
    "fox",
    "news",
    "home depot",
    "homedepot",
    "squarespace",
)
NOUL_THRESHOLD: Final[float] = 0.5
PAIRED_MARGIN_FLOOR: Final[float] = 0.10

_VISUAL_INSTRUCTIONS: Final[str] = "Is the website in this screenshot Fox News?"
_VISUAL_CRITERIA: Final[dict[str, str]] = {
    "true": ("Fox News site chrome or branding is clearly visible in the screenshot."),
    "false": "It is not.",
}


@dataclass(frozen=True, slots=True)
class VisualControl:
    """One matched condition for the visually dependent question.

    Attributes:
        unique_data_id (str): Row under test.
        visual_family (str): Visual family of that row.
        condition (str): One of ``CONDITIONS``.
        state (str): Prompt state; always the family-neutral constant.
        image_file (str | None): Screenshot to attach, ``None`` when omitted.
        image_unique_data_id (str | None): Row the screenshot came from.
        expected (bool | None): Gold polarity, ``None`` when nothing is provable.
        counts_as_semantic_hit (bool): Whether a match may score.

    Examples:
        ```python
        from typevet.evaluation.datasets.psai_vision_controls import VisualControl

        control = VisualControl(
            unique_data_id="row",
            visual_family="fox_news",
            condition="omitted",
            state="neutral",
            image_file=None,
            image_unique_data_id=None,
            expected=None,
            counts_as_semantic_hit=False,
        )
        assert control.expected is None
        ```
    """

    unique_data_id: str
    visual_family: str
    condition: str
    state: str
    image_file: str | None
    image_unique_data_id: str | None
    expected: bool | None
    counts_as_semantic_hit: bool


def visual_question() -> Noul:
    """Return the manual visually dependent Noul.

    The text names Fox News because that is the question. It never says which
    row or family the attached screenshot belongs to.

    Returns:
        Noul with ``true`` and ``false`` criteria.
    """
    return Noul(instructions=_VISUAL_INSTRUCTIONS, criteria=dict(_VISUAL_CRITERIA))


def annotation_questions() -> dict[str, Question]:
    """Return the Hub annotation-backed questions.

    Returns:
        ``category`` Choice and ``requires_login`` Noul, with Hub instructions.
    """
    props = METADATA_DECISIONS_SCHEMA["properties"]
    return {
        "category": Choice(
            criteria=dict.fromkeys(CATEGORY_LABELS, None),
            instructions=props["category"]["instructions"],
        ),
        PRIMARY_NOUL_NAME: Noul(instructions=props[PRIMARY_NOUL_NAME]["instructions"]),
    }


def vision_questions() -> dict[str, Question]:
    """Return every question the smoke scores.

    Returns:
        The annotation-backed pair plus the manual visual Noul.
    """
    questions = annotation_questions()
    questions[VISUAL_QUESTION_NAME] = visual_question()
    return questions


def question_provenance() -> dict[str, str]:
    """Return the gold provenance of each question.

    Returns:
        Question name to ``annotation`` or ``manual_visual``.
    """
    return {
        "category": ANNOTATION_PROVENANCE,
        PRIMARY_NOUL_NAME: ANNOTATION_PROVENANCE,
        VISUAL_QUESTION_NAME: MANUAL_VISUAL_PROVENANCE,
    }


def annotation_state(example: PsaiVisionExample) -> dict[str, str]:
    """Build the annotation-leg state for one row.

    Args:
        example: Row whose task text to evaluate.

    Returns:
        Mapping with a single ``task_name`` key.

    Raises:
        ValueError: When the state would carry a banned leakage key.
    """
    state = psai_state(example.task_name)
    validate_task_state(state)
    return state


def validate_no_family_leak(text: str) -> None:
    """Reject prompt text that names a visual family.

    Args:
        text: Candidate state text for the visual leg.

    Raises:
        ValueError: When ``text`` holds a term from ``FAMILY_LEAK_TERMS``, which
            would let the model answer the visual question without the image.
    """
    lowered = text.lower()
    found = [term for term in FAMILY_LEAK_TERMS if term in lowered]
    if found:
        names = ", ".join(found)
        msg = (
            f"visual-leg text would leak the visual family; remove: {names}. "
            "A text-derivable answer voids the image controls."
        )
        raise ValueError(msg)


def _swap_donors(
    examples: Sequence[PsaiVisionExample],
) -> dict[str, PsaiVisionExample]:
    """Pair each row with a deterministic opposite-family donor.

    Args:
        examples: Rows to pair.

    Returns:
        ``unique_data_id`` to the opposite-family row that donates its image.

    Raises:
        ValueError: When either visual family has no row to donate.
    """
    by_family = {
        family: sorted(
            (e for e in examples if e.visual_family == family),
            key=lambda e: e.stream_order,
        )
        for family in VISUAL_FAMILIES
    }
    for family, rows in by_family.items():
        if not rows:
            msg = (
                f"the swap control needs at least one {family!r} row; "
                f"got none among {len(examples)} rows"
            )
            raise ValueError(msg)
    opposite = {FOX_FAMILY: NON_FOX_FAMILY, NON_FOX_FAMILY: FOX_FAMILY}
    donors: dict[str, PsaiVisionExample] = {}
    for family, rows in by_family.items():
        pool = by_family[opposite[family]]
        for position, row in enumerate(rows):
            donors[row.unique_data_id] = pool[position % len(pool)]
    return donors


def control_matrix(
    examples: Sequence[PsaiVisionExample],
) -> tuple[VisualControl, ...]:
    """Build matched present / omitted / swapped controls for every row.

    Args:
        examples: Rows to evaluate; both visual families must appear.

    Returns:
        Controls grouped by row, in ``CONDITIONS`` order.

    Raises:
        ValueError: When either visual family has no row, so no swap donor
            exists.
    """
    donors = _swap_donors(examples)
    controls: list[VisualControl] = []
    for example in examples:
        donor = donors[example.unique_data_id]
        controls.extend(
            (
                VisualControl(
                    unique_data_id=example.unique_data_id,
                    visual_family=example.visual_family,
                    condition=CONDITION_PRESENT,
                    state=VISUAL_CONTROL_STATE,
                    image_file=example.screenshot.file_name,
                    image_unique_data_id=example.unique_data_id,
                    expected=example.shows_fox_news_chrome,
                    counts_as_semantic_hit=True,
                ),
                VisualControl(
                    unique_data_id=example.unique_data_id,
                    visual_family=example.visual_family,
                    condition=CONDITION_OMITTED,
                    state=VISUAL_CONTROL_STATE,
                    image_file=None,
                    image_unique_data_id=None,
                    expected=None,
                    counts_as_semantic_hit=False,
                ),
                VisualControl(
                    unique_data_id=example.unique_data_id,
                    visual_family=example.visual_family,
                    condition=CONDITION_SWAPPED,
                    state=VISUAL_CONTROL_STATE,
                    image_file=donor.screenshot.file_name,
                    image_unique_data_id=donor.unique_data_id,
                    expected=donor.shows_fox_news_chrome,
                    counts_as_semantic_hit=True,
                ),
            )
        )
    return tuple(controls)


def noul_polarity(probability: float, threshold: float = NOUL_THRESHOLD) -> bool:
    """Read a Noul probability as a boolean answer.

    Args:
        probability: Probability the scorer gave the ``true`` outcome.
        threshold: Cut at or above which the answer reads true.

    Returns:
        ``True`` when ``probability >= threshold``.
    """
    return probability >= threshold


def semantic_hit(
    control: VisualControl,
    probability: float,
    threshold: float = NOUL_THRESHOLD,
) -> bool:
    """Score one control against the contract semantic rule.

    Present must match the row's own gold. Swapped must match the donor family,
    so the polarity flips. Omitted never scores, whatever it answered.

    Args:
        control: Control row under test.
        probability: Scored probability of the ``true`` outcome.
        threshold: Polarity cut passed to ``noul_polarity``.

    Returns:
        ``True`` only for a counted control whose polarity matches ``expected``.
    """
    if not control.counts_as_semantic_hit or control.expected is None:
        return False
    return noul_polarity(probability, threshold) is control.expected


@dataclass(frozen=True, slots=True)
class PairedOrdering:
    """The present/swapped pair for one row, compared image against image.

    Both members carry byte-identical text, so the probability difference is
    attributable to the pixels alone. This needs no calibration assumption,
    unlike reading one probability against a fixed threshold.

    Attributes:
        unique_data_id (str): Row the pair belongs to.
        fox_image_id (str): Row that supplied the Fox News screenshot.
        fox_probability (float): Scored ``true`` probability for that image.
        non_fox_image_id (str): Row that supplied the non-Fox screenshot.
        non_fox_probability (float): Scored ``true`` probability for that image.
        margin (float): ``fox_probability - non_fox_probability``.
        ordered (bool): Whether ``margin`` clears the floor.

    Examples:
        ```python
        from typevet.evaluation.datasets.psai_vision_controls import PairedOrdering

        pair = PairedOrdering(
            unique_data_id="row",
            fox_image_id="fox",
            fox_probability=0.92,
            non_fox_image_id="shop",
            non_fox_probability=0.09,
            margin=0.83,
            ordered=True,
        )
        assert pair.ordered
        ```
    """

    unique_data_id: str
    fox_image_id: str
    fox_probability: float
    non_fox_image_id: str
    non_fox_probability: float
    margin: float
    ordered: bool


def paired_image_ordering(
    controls: Sequence[VisualControl],
    probabilities: Mapping[tuple[str, str], float],
    floor: float = PAIRED_MARGIN_FLOOR,
) -> tuple[PairedOrdering, ...]:
    """Compare each row's two imaged controls against one another.

    The present and swapped controls of one row always hold opposite families,
    because the swap donor comes from the other family. Requiring the Fox image
    to outscore the non-Fox image tests conditioning without assuming the model
    is calibrated at any particular cut.

    Args:
        controls: Control rows that were run.
        probabilities: ``(unique_data_id, condition)`` to scored probability.
        floor: Smallest margin that still counts as ordered.

    Returns:
        One ``PairedOrdering`` per row, in row order.

    Raises:
        KeyError: When an imaged control has no scored probability.
        ValueError: When a row does not hold one Fox and one non-Fox image.
    """
    pairs: list[PairedOrdering] = []
    for uid in dict.fromkeys(control.unique_data_id for control in controls):
        imaged = [
            control
            for control in controls
            if control.unique_data_id == uid and control.image_unique_data_id
        ]
        fox = [control for control in imaged if control.expected is True]
        non_fox = [control for control in imaged if control.expected is False]
        if len(fox) != 1 or len(non_fox) != 1:
            msg = (
                f"row {uid!r} needs one Fox and one non-Fox image to pair; got "
                f"{len(fox)} Fox and {len(non_fox)} non-Fox"
            )
            raise ValueError(msg)
        fox_probability = probabilities[(uid, fox[0].condition)]
        non_fox_probability = probabilities[(uid, non_fox[0].condition)]
        margin = fox_probability - non_fox_probability
        pairs.append(
            PairedOrdering(
                unique_data_id=uid,
                fox_image_id=str(fox[0].image_unique_data_id),
                fox_probability=fox_probability,
                non_fox_image_id=str(non_fox[0].image_unique_data_id),
                non_fox_probability=non_fox_probability,
                margin=margin,
                ordered=margin >= floor,
            )
        )
    return tuple(pairs)


def control_summary(
    controls: Sequence[VisualControl],
    probabilities: Mapping[tuple[str, str], float],
    threshold: float = NOUL_THRESHOLD,
) -> dict[str, int]:
    """Count counted controls and semantic hits for a receipt.

    Args:
        controls: Control rows that were run.
        probabilities: ``(unique_data_id, condition)`` to scored probability.
        threshold: Polarity cut passed to ``semantic_hit``.

    Returns:
        Counts under ``counted``, ``hits`` and ``uncounted``.

    Raises:
        KeyError: When a control has no scored probability.
    """
    counted = 0
    hits = 0
    for control in controls:
        probability = probabilities[(control.unique_data_id, control.condition)]
        if not control.counts_as_semantic_hit:
            continue
        counted += 1
        if semantic_hit(control, probability, threshold):
            hits += 1
    return {
        "counted": counted,
        "hits": hits,
        "uncounted": len(controls) - counted,
    }
