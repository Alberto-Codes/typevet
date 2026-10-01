"""The named text parts of a Noul that a wording run may evolve (#363).

A wording candidate is a mapping of part name to text. The part names are
the identifiers ``instructions``, ``criteria_true`` and ``criteria_false``,
because gepa-adk refuses a component name that is not an identifier.
``seed_mapping`` builds the full mapping from a seed ``Noul``. A seed
without ``criteria`` has the ``instructions`` part only. ``noul_from_parts``
builds the question that a call sends. ``WordingParts`` holds the evolved
selection and the seed and evolved mappings of one run. Its construction
fails when a frozen part changed. ``artifact_parts`` reads them back from an
evolution artifact.

Every refusal names the part or the question type, never a text, so an
error message cannot leak a wording into a log.

This module does not import judgevet: the caller gives the seed ``Noul``.

Attributes:
    INSTRUCTIONS (str): The part name of the question text.
    CRITERIA_TRUE (str): The part name of the ``true`` criterion.
    CRITERIA_FALSE (str): The part name of the ``false`` criterion.
    PART_NAMES (tuple[str, ...]): Every part name, in render order.
    PART_ROLES (Mapping[str, str]): The role of each part, for the reflector.

Examples:
    ```python
    from judgevet.domain.questions import Noul

    seed = Noul(instructions="Is it a scam?", criteria={"true": "Yes", "false": "No"})
    parts = seed_mapping(seed)
    # {"instructions": "Is it a scam?", "criteria_true": "Yes", "criteria_false": "No"}
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport that renders the parts
    - [typevet_evals.wording.digests][]: the receipt digests of the parts
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol, Self

__all__ = [
    "CRITERIA_FALSE",
    "CRITERIA_TRUE",
    "INSTRUCTIONS",
    "PART_NAMES",
    "PART_ROLES",
    "SeedNoul",
    "WordingParts",
    "artifact_parts",
    "check_parts",
    "check_selection",
    "noul_from_parts",
    "receipt_parts",
    "seed_mapping",
]

INSTRUCTIONS: Final[str] = "instructions"
CRITERIA_TRUE: Final[str] = "criteria_true"
CRITERIA_FALSE: Final[str] = "criteria_false"
PART_NAMES: Final[tuple[str, ...]] = (INSTRUCTIONS, CRITERIA_TRUE, CRITERIA_FALSE)
PART_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        INSTRUCTIONS: "the yes/no question the judge answers about the message",
        CRITERIA_TRUE: "what a yes answer means",
        CRITERIA_FALSE: "what a no answer means",
    }
)
_CRITERIA_KEYS: Final[dict[str, str]] = {"true": CRITERIA_TRUE, "false": CRITERIA_FALSE}


class SeedNoul(Protocol):
    """The seed judgevet ``Noul`` shape: keyword construction and two fields.

    Attributes:
        instructions (object): The seed wording.
        criteria (Any): The ``true`` and ``false`` descriptions, or None.

    Examples:
        ```python
        from judgevet.domain.questions import Noul

        seed: SeedNoul = Noul(instructions="Is this message a scam?")
        ```
    """

    def __init__(self, *, instructions: Any = None, criteria: Any = None) -> None:
        """Build a question from its wording and criteria."""

    @property
    def instructions(self) -> object:
        """Return the wording."""
        ...

    @property
    def criteria(self) -> Any:
        """Return the criteria."""
        ...


def seed_mapping(seed: SeedNoul) -> dict[str, str]:
    """Return the full part mapping of a seed ``Noul``.

    Args:
        seed: The seed question.

    Returns:
        ``instructions``, plus ``criteria_true`` and ``criteria_false`` when
        the seed has criteria.

    Raises:
        ValueError: If the seed is not a ``Noul``, or its criteria do not
            hold exactly a ``true`` and a ``false`` key.
        TypeError: If the instructions or a criterion is not text.
    """
    kind = type(seed).__name__
    if kind != "Noul":
        msg = f"a {kind} seed has no wording parts yet; only a Noul seed does"
        raise ValueError(msg)
    if not isinstance(seed.instructions, str):
        raise TypeError("the seed instructions must be text")
    parts = {INSTRUCTIONS: seed.instructions}
    criteria = seed.criteria
    if criteria is None:
        return parts
    if not isinstance(criteria, Mapping) or set(criteria) != set(_CRITERIA_KEYS):
        raise ValueError("the seed criteria must hold only 'true' and 'false' texts")
    for label, name in _CRITERIA_KEYS.items():
        if not isinstance(criteria[label], str):
            msg = f"the seed criteria {label!r} must be text"
            raise TypeError(msg)
        parts[name] = criteria[label]
    return parts


def check_parts(mapping: Mapping[str, object], seed_parts: Mapping[str, str]) -> None:
    """Refuse a mapping whose part names differ from the seed's.

    Args:
        mapping: The mapping to check.
        seed_parts: The seed's full mapping.

    Raises:
        ValueError: If a key is not a part of the seed, or a seed part is
            missing. The message holds the key only.
        TypeError: If a value is not text. The message holds the key only.
    """
    for key in mapping:
        if key not in seed_parts:
            known = "a part of the seed" if key in PART_NAMES else "a wording part"
            msg = f"{key!r} is not {known}"
            raise ValueError(msg)
        if not isinstance(mapping[key], str):
            msg = f"the wording part {key!r} must be text"
            raise TypeError(msg)
    for key in seed_parts:
        if key not in mapping:
            msg = f"the wording mapping lacks the part {key!r}"
            raise ValueError(msg)


def check_selection(
    components: Iterable[str], seed_parts: Mapping[str, str]
) -> tuple[str, ...]:
    """Return the evolved selection after checking it against the seed.

    Args:
        components: The part names to evolve.
        seed_parts: The seed's full mapping.

    Returns:
        The selection, in the given order.

    Raises:
        ValueError: If the selection is empty, names a part twice, or names
            a part the seed lacks.
    """
    selection = tuple(components)
    if not selection:
        raise ValueError("select at least one wording part")
    for name in selection:
        if name not in seed_parts:
            msg = f"{name!r} is not a part of the seed"
            raise ValueError(msg)
        if selection.count(name) > 1:
            msg = f"{name!r} is selected twice"
            raise ValueError(msg)
    return selection


def noul_from_parts(seed: SeedNoul, parts: Mapping[str, str]) -> SeedNoul:
    """Return a question of the seed's type built from a full part mapping.

    Args:
        seed: The seed question; only its type is used.
        parts: A full mapping, as ``seed_mapping`` gives.

    Returns:
        The question with ``parts["instructions"]``, and with ``true`` and
        ``false`` criteria when the mapping has them.
    """
    criteria = None
    if CRITERIA_TRUE in parts:
        criteria = {"true": parts[CRITERIA_TRUE], "false": parts[CRITERIA_FALSE]}
    return type(seed)(instructions=parts[INSTRUCTIONS], criteria=criteria)


@dataclass(frozen=True, slots=True)
class WordingParts:
    """The evolved selection and the seed and evolved mappings of one run.

    Construction copies both mappings, so a later change to a caller's
    mapping does not change the parts. It checks the selection and the
    evolved mapping against the seed, and fails when a frozen part (one not
    in ``components``) changed.

    Attributes:
        components (tuple[str, ...]): The part names the run evolved.
        seed (Mapping[str, str]): The seed's full mapping.
        evolved (Mapping[str, str]): The evolved full mapping.

    Examples:
        ```python
        parts = WordingParts.from_texts("Is it a scam?", "Does it ask for money?")
        parts.evolved_text  # "Does it ask for money?"
        ```
    """

    components: tuple[str, ...]
    seed: Mapping[str, str]
    evolved: Mapping[str, str]

    def __post_init__(self) -> None:
        """Copy both mappings, then check the selection, names and frozen parts.

        Raises:
            ValueError: If the selection or a mapping is not valid, or a
                frozen part changed. The message names the part only.
            TypeError: If a part is not text.
        """
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "seed", dict(self.seed))
        object.__setattr__(self, "evolved", dict(self.evolved))
        check_parts(self.seed, self.seed)
        check_selection(self.components, self.seed)
        check_parts(self.evolved, self.seed)
        for name, text in self.seed.items():
            if name not in self.components and self.evolved[name] != text:
                msg = f"frozen part {name!r} changed during the run"
                raise ValueError(msg)

    @classmethod
    def from_texts(cls, seed_text: str, evolved_text: str) -> Self:
        """Return the parts of an ``instructions``-only run.

        Args:
            seed_text: The seed instructions.
            evolved_text: The evolved instructions.

        Returns:
            Parts that select and hold ``instructions`` only.
        """
        return cls(
            (INSTRUCTIONS,), {INSTRUCTIONS: seed_text}, {INSTRUCTIONS: evolved_text}
        )

    @classmethod
    def instructions_only(cls, seed: SeedNoul, evolved_text: str) -> Self:
        """Return the parts of an ``instructions``-only run of any seed.

        Args:
            seed: The seed ``Noul``, with or without criteria.
            evolved_text: The evolved instructions.

        Returns:
            Parts that select ``instructions`` and hold the seed's full
            mapping, with ``evolved_text`` as the evolved instructions.
        """
        seed_parts = seed_mapping(seed)
        evolved = seed_parts | {INSTRUCTIONS: evolved_text}
        return cls((INSTRUCTIONS,), seed_parts, evolved)

    @property
    def seed_text(self) -> str:
        """Return the seed ``instructions`` text.

        Returns:
            ``seed["instructions"]``.
        """
        return self.seed[INSTRUCTIONS]

    @property
    def evolved_text(self) -> str:
        """Return the evolved ``instructions`` text.

        Returns:
            ``evolved["instructions"]``.
        """
        return self.evolved[INSTRUCTIONS]

    def check_texts(self, seed_text: str, evolved_text: str) -> None:
        """Refuse texts that are not this run's ``instructions`` texts.

        Args:
            seed_text: The seed text a caller gives.
            evolved_text: The evolved text a caller gives.

        Raises:
            ValueError: If either text differs; the message names the field only.
        """
        for field, given, held in (
            ("seed_text", seed_text, self.seed_text),
            ("evolved_text", evolved_text, self.evolved_text),
        ):
            if given != held:
                msg = f"{field} is not the {INSTRUCTIONS!r} part of the parts"
                raise ValueError(msg)


def receipt_parts(
    seed_text: str, evolved_text: str, parts: WordingParts | None
) -> WordingParts:
    """Return the parts a receipt records.

    Args:
        seed_text: The seed ``instructions`` text.
        evolved_text: The evolved ``instructions`` text.
        parts: The run's parts, or None for an ``instructions``-only run.

    Returns:
        ``parts``, or ``WordingParts.from_texts(seed_text, evolved_text)``.

    Raises:
        ValueError: If ``parts`` holds other ``instructions`` texts.
    """
    if parts is None:
        return WordingParts.from_texts(seed_text, evolved_text)
    parts.check_texts(seed_text, evolved_text)
    return parts


def artifact_parts(
    artifact: Mapping[str, Any], seed_parts: Mapping[str, str]
) -> WordingParts:
    """Return the parts of an evolution artifact, checked against the seed.

    An artifact written before #363 has no ``evolved_parts``. Its evolved
    ``instructions`` text then goes onto the seed's parts, so both arms send
    the seed's criteria, as ``WordingParts.instructions_only`` gives.

    Args:
        artifact: The evolution artifact.
        seed_parts: The full mapping of the seed the caller scores.

    Returns:
        The artifact's selection and evolved mapping over ``seed_parts``.

    Raises:
        ValueError: If the artifact's seed or evolved ``instructions`` text,
            its ``seed_parts`` or its ``components`` do not agree with the
            seed and the part names, or a frozen part changed. The message
            names the field only, never a value.
        TypeError: If ``evolved_parts`` is not a mapping.
    """
    if artifact.get("seed_text") != seed_parts[INSTRUCTIONS]:
        raise ValueError("the artifact seed_text is not the seed's instructions part")
    if "evolved_parts" not in artifact:
        evolved = dict(seed_parts) | {INSTRUCTIONS: str(artifact["evolved_text"])}
        return WordingParts((INSTRUCTIONS,), dict(seed_parts), evolved)
    components = artifact.get("components")
    if not isinstance(components, list) or not all(
        name in PART_NAMES for name in components
    ):
        raise ValueError("the artifact components name a part outside the Noul parts")
    if artifact.get("seed_parts") != dict(seed_parts):
        raise ValueError("the artifact seed_parts are not the seed's parts")
    evolved = artifact["evolved_parts"]
    if not isinstance(evolved, Mapping):
        raise TypeError("the artifact evolved_parts must be a mapping")
    parts = WordingParts(tuple(components), dict(seed_parts), dict(evolved))
    if artifact.get("evolved_text") != parts.evolved_text:
        raise ValueError("the artifact evolved_text is not its instructions part")
    return parts
