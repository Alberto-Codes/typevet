"""Vendored PSAI first-screenshot fixtures for the vision smoke ([#154][i154]).

The metadata loader in [typevet.evaluation.datasets.psai][] strips screenshots.
This module reads the fixed set vendored under
``tests/fixtures/psai/vision_smoke/``: one unmodified first screenshot per
``unique_data_id``, plus a manifest that keeps MIT provenance and holds every
gold answer under ``expected``.

Two gold provenances live side by side. ``category`` and ``requires_login``
come from Hub annotations. ``shows_fox_news_chrome`` is manual visual labelling
and says so, because no Hub field describes the pixels.

[i154]: https://github.com/Alberto-Codes/typevet/issues/154

Examples:
    Load the vendored manifest and split it by visual family:

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.psai_vision import (
        FOX_FAMILY,
        load_vision_smoke,
    )

    root = Path("tests/fixtures/psai/vision_smoke")
    fixture = load_vision_smoke((root / "manifest.json").read_text())
    assert len(fixture.by_family(FOX_FAMILY)) >= 2
    ```

See Also:
    - [typevet.evaluation.datasets.psai_vision_controls][]: questions and controls
    - [typevet.evaluation.datasets.psai][]: metadata-only loader and field map
    - docs/how-to/run-the-psai-vision-smoke.md: the recipe

Attributes:
    MANIFEST_VERSION (int): Manifest shape this module reads.
    VISUAL_QUESTION_NAME (str): Name of the manual visually dependent question.
    FOX_FAMILY (str): Visual family label for Fox News screenshots.
    NON_FOX_FAMILY (str): Visual family label for every other screenshot.
    VISUAL_FAMILIES (tuple[str, ...]): Both accepted family labels.
    ANNOTATION_PROVENANCE (str): Gold provenance tag for Hub-backed fields.
    MANUAL_VISUAL_PROVENANCE (str): Gold provenance tag for manual visual labels.
    GOLD_FIELDS (tuple[str, ...]): Every field the manifest may score.
    ACCEPTED_UNIQUE_DATA_IDS (tuple[str, ...]): Fixed ids in stream order.
    MINIMUM_ROWS_PER_FAMILY (int): Rows each family needs for a swap control.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from typevet.domain.errors import ScoringValidationError
from typevet.domain.media import ImageInput
from typevet.evaluation.datasets.psai_schema import (
    DATASET_ID,
    DATASET_LICENSE,
    PRIMARY_NOUL_NAME,
    SPLIT,
)

MANIFEST_VERSION: Final[int] = 1
VISUAL_QUESTION_NAME: Final[str] = "shows_fox_news_chrome"
FOX_FAMILY: Final[str] = "fox_news"
NON_FOX_FAMILY: Final[str] = "non_fox"
VISUAL_FAMILIES: Final[tuple[str, ...]] = (FOX_FAMILY, NON_FOX_FAMILY)
ANNOTATION_PROVENANCE: Final[str] = "annotation"
MANUAL_VISUAL_PROVENANCE: Final[str] = "manual_visual"
GOLD_FIELDS: Final[tuple[str, ...]] = (
    "category",
    PRIMARY_NOUL_NAME,
    VISUAL_QUESTION_NAME,
)
ACCEPTED_UNIQUE_DATA_IDS: Final[tuple[str, ...]] = (
    "cmcc8u6yc00va1p1ydsdu52zy",
    "cmcc8u6yc00v91p1yw2eruz95",
    "cmcc8u6yc00vm1p1yhjl1u0bf",
    "cmcc8u6yd00wv1p1yy8guorre",
    "cmcc8u6yd00wr1p1yj7aot3ae",
)
MINIMUM_ROWS_PER_FAMILY: Final[int] = 2
MANIFEST_FILE_NAME: Final[str] = "manifest.json"

_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "manifest_version",
    "dataset_id",
    "license",
    "split",
    "source",
    "screenshot_index",
    "gold_provenance",
    "rows",
)
_ROW_KEYS: Final[tuple[str, ...]] = (
    "unique_data_id",
    "stream_order",
    "task_name",
    "visual_family",
    "screenshot",
    "expected",
)
_SCREENSHOT_KEYS: Final[tuple[str, ...]] = (
    "file_name",
    "index",
    "source_path",
    "mime_type",
    "sha256",
    "byte_count",
    "width",
    "height",
)
_GOLD_PROVENANCES: Final[frozenset[str]] = frozenset(
    {ANNOTATION_PROVENANCE, MANUAL_VISUAL_PROVENANCE}
)


@dataclass(frozen=True, slots=True)
class VisionScreenshot:
    """One vendored PNG and the bytes it must still hash to.

    Attributes:
        file_name (str): Name beside the manifest, ``<unique_data_id>.png``.
        index (int): Screenshot position inside the Hub row (``0``).
        source_path (str): Original Hub ``screenshots[index].path`` value.
        mime_type (str): Always ``image/png`` for this fixture set.
        sha256 (str): Digest of the vendored bytes.
        byte_count (int): Size of the vendored bytes.
        width (int): Pixel width read from the PNG header.
        height (int): Pixel height read from the PNG header.

    Examples:
        ```python
        from typevet.evaluation.datasets.psai_vision import VisionScreenshot

        VisionScreenshot(
            file_name="row.png",
            index=0,
            source_path="tmp.png",
            mime_type="image/png",
            sha256="00",
            byte_count=1,
            width=1280,
            height=720,
        )
        ```
    """

    file_name: str
    index: int
    source_path: str
    mime_type: str
    sha256: str
    byte_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PsaiVisionExample:
    """One PSAI row with a screenshot and its two kinds of gold.

    Attributes:
        unique_data_id (str): Hub dedupe key, preserved from the corpus.
        stream_order (int): Position in the train stream the row came from.
        task_name (str): Task text. Names the site, so the visual leg drops it.
        visual_family (str): ``fox_news`` or ``non_fox``.
        screenshot (VisionScreenshot): The vendored first screenshot.
        category (str): Annotation-backed Choice gold.
        requires_login (bool): Annotation-backed Noul gold.
        shows_fox_news_chrome (bool): Manual visual Noul gold.

    Examples:
        ```python
        from pathlib import Path

        from typevet.evaluation.datasets.psai_vision import load_vision_smoke

        root = Path("tests/fixtures/psai/vision_smoke")
        example = load_vision_smoke((root / "manifest.json").read_text()).examples[0]
        assert example.expected()["category"] == "BROWSER_TASK"
        ```
    """

    unique_data_id: str
    stream_order: int
    task_name: str
    visual_family: str
    screenshot: VisionScreenshot
    category: str
    requires_login: bool
    shows_fox_news_chrome: bool

    @property
    def is_fox(self) -> bool:
        """Report whether this row belongs to the Fox News visual family.

        Returns:
            ``True`` when ``visual_family`` is ``fox_news``.
        """
        return self.visual_family == FOX_FAMILY

    def expected(self) -> dict[str, Any]:
        """Return every gold answer for this row.

        Gold lives here and nowhere else. Never merge this into a prompt state.

        Returns:
            Mapping of question name to gold answer.
        """
        return {
            "category": self.category,
            PRIMARY_NOUL_NAME: self.requires_login,
            VISUAL_QUESTION_NAME: self.shows_fox_news_chrome,
        }

    def provenance(self, fixture: VisionSmokeFixture) -> dict[str, Any]:
        """Return corpus provenance for this row.

        Args:
            fixture: Loaded fixture set holding the dataset-level fields.

        Returns:
            Mapping with dataset id, licence, split, source, id and shot index.
        """
        return {
            "dataset_id": fixture.dataset_id,
            "license": fixture.license,
            "split": fixture.split,
            "source": fixture.source,
            "unique_data_id": self.unique_data_id,
            "stream_order": self.stream_order,
            "screenshot_index": self.screenshot.index,
            "screenshot_sha256": self.screenshot.sha256,
        }


@dataclass(frozen=True, slots=True)
class VisionSmokeFixture:
    """The vendored fixture set and its corpus-level provenance.

    Attributes:
        dataset_id (str): Hub dataset id.
        license (str): Corpus licence, ``MIT``.
        split (str): Hub split the rows came from.
        source (str): Corpus id used in eval manifests.
        screenshot_index (int): Screenshot position every row vendored.
        gold_provenance (Mapping[str, str]): Question name to gold provenance.
        examples (tuple[PsaiVisionExample, ...]): Rows in stream order.

    Examples:
        ```python
        from pathlib import Path

        from typevet.evaluation.datasets.psai_vision import load_vision_smoke

        root = Path("tests/fixtures/psai/vision_smoke")
        fixture = load_vision_smoke((root / "manifest.json").read_text())
        assert fixture.license == "MIT"
        ```
    """

    dataset_id: str
    license: str
    split: str
    source: str
    screenshot_index: int
    gold_provenance: Mapping[str, str]
    examples: tuple[PsaiVisionExample, ...]

    def by_family(self, family: str) -> tuple[PsaiVisionExample, ...]:
        """Return the rows of one visual family in stream order.

        Args:
            family: ``fox_news`` or ``non_fox``.

        Returns:
            Matching rows, empty when the family has none.

        Raises:
            ValueError: When ``family`` is not a known visual_family label.
        """
        if family not in VISUAL_FAMILIES:
            allowed = ", ".join(VISUAL_FAMILIES)
            msg = f"unknown visual_family {family!r}; allowed: {allowed}"
            raise ValueError(msg)
        return tuple(e for e in self.examples if e.visual_family == family)


def vision_smoke_manifest_path(root: Path) -> Path:
    """Return the manifest path inside a vendored fixture directory.

    Args:
        root: Directory holding the PNGs and the manifest.

    Returns:
        Path to ``manifest.json`` under ``root``.
    """
    return root / MANIFEST_FILE_NAME


def _as_object(raw: Any, where: str) -> Mapping[str, Any]:
    """Return ``raw`` as a mapping, or reject the manifest.

    A wrong JSON type is a manifest defect, so every failure here is a
    ``ValueError`` like the rest of the loader.

    Args:
        raw: Parsed JSON value.
        where: Location name used in the message.

    Returns:
        ``raw`` when it is a mapping.

    Raises:
        ValueError: When ``raw`` is not a JSON object.
    """
    if isinstance(raw, Mapping):
        return raw
    msg = f"PSAI vision {where} must be an object"
    raise ValueError(msg)


def _as_array(raw: Any, where: str) -> Sequence[Any]:
    """Return ``raw`` as a non-string sequence, or reject the manifest.

    Args:
        raw: Parsed JSON value.
        where: Location name used in the message.

    Returns:
        ``raw`` when it is a non-string sequence.

    Raises:
        ValueError: When ``raw`` is not a JSON array.
    """
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return raw
    msg = f"PSAI vision {where} must be an array"
    raise ValueError(msg)


def _require_keys(raw: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        names = ", ".join(missing)
        msg = f"PSAI vision {where} missing required keys: {names}"
        raise ValueError(msg)


def _parse_screenshot(raw: Any, uid: str) -> VisionScreenshot:
    raw = _as_object(raw, f"row {uid!r} screenshot")
    _require_keys(raw, _SCREENSHOT_KEYS, f"row {uid!r} screenshot")
    return VisionScreenshot(
        file_name=str(raw["file_name"]),
        index=int(raw["index"]),
        source_path=str(raw["source_path"]),
        mime_type=str(raw["mime_type"]),
        sha256=str(raw["sha256"]),
        byte_count=int(raw["byte_count"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
    )


def _parse_expected(raw: Any, uid: str, family: str) -> dict[str, Any]:
    raw = _as_object(raw, f"row {uid!r} expected")
    if set(raw) != set(GOLD_FIELDS):
        wanted = ", ".join(sorted(GOLD_FIELDS))
        got = ", ".join(sorted(str(key) for key in raw))
        msg = f"PSAI vision row {uid!r} expected keys must be {wanted}; got {got}"
        raise ValueError(msg)
    visual = bool(raw[VISUAL_QUESTION_NAME])
    if visual is not (family == FOX_FAMILY):
        msg = (
            f"PSAI vision row {uid!r} visual_family {family!r} disagrees with "
            f"{VISUAL_QUESTION_NAME}={visual!r}"
        )
        raise ValueError(msg)
    return dict(raw)


def _parse_row(raw: Any) -> PsaiVisionExample:
    raw = _as_object(raw, "manifest rows must be objects; entry")
    _require_keys(raw, _ROW_KEYS, "row")
    uid = str(raw["unique_data_id"])
    outside = sorted(set(raw).intersection(GOLD_FIELDS))
    if outside:
        names = ", ".join(outside)
        msg = f"PSAI vision row {uid!r} holds gold outside expected: {names}"
        raise ValueError(msg)
    family = str(raw["visual_family"])
    if family not in VISUAL_FAMILIES:
        allowed = ", ".join(VISUAL_FAMILIES)
        msg = f"PSAI vision row {uid!r} visual_family {family!r} not in {allowed}"
        raise ValueError(msg)
    expected = _parse_expected(raw["expected"], uid, family)
    return PsaiVisionExample(
        unique_data_id=uid,
        stream_order=int(raw["stream_order"]),
        task_name=str(raw["task_name"]),
        visual_family=family,
        screenshot=_parse_screenshot(raw["screenshot"], uid),
        category=str(expected["category"]),
        requires_login=bool(expected[PRIMARY_NOUL_NAME]),
        shows_fox_news_chrome=bool(expected[VISUAL_QUESTION_NAME]),
    )


def _parse_gold_provenance(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(GOLD_FIELDS):
        wanted = ", ".join(sorted(GOLD_FIELDS))
        msg = f"PSAI vision gold_provenance must name exactly {wanted}"
        raise ValueError(msg)
    provenance = {str(key): str(value) for key, value in raw.items()}
    unknown = sorted(set(provenance.values()) - _GOLD_PROVENANCES)
    if unknown:
        allowed = ", ".join(sorted(_GOLD_PROVENANCES))
        names = ", ".join(unknown)
        msg = f"PSAI vision gold_provenance values {names} not in {allowed}"
        raise ValueError(msg)
    if provenance[VISUAL_QUESTION_NAME] != MANUAL_VISUAL_PROVENANCE:
        msg = (
            f"PSAI vision gold_provenance must mark {VISUAL_QUESTION_NAME} "
            f"as {MANUAL_VISUAL_PROVENANCE}"
        )
        raise ValueError(msg)
    return provenance


def _validate_rows(examples: Sequence[PsaiVisionExample]) -> None:
    seen: set[str] = set()
    for example in examples:
        if example.unique_data_id in seen:
            msg = f"PSAI vision manifest has duplicate id {example.unique_data_id!r}"
            raise ValueError(msg)
        seen.add(example.unique_data_id)
    for family in VISUAL_FAMILIES:
        count = sum(1 for e in examples if e.visual_family == family)
        if count < MINIMUM_ROWS_PER_FAMILY:
            msg = (
                f"PSAI vision manifest needs at least {MINIMUM_ROWS_PER_FAMILY} "
                f"{family!r} rows for a swap control; got {count}"
            )
            raise ValueError(msg)


def load_vision_smoke(manifest_text: str) -> VisionSmokeFixture:
    """Parse and validate a vendored vision smoke manifest.

    Args:
        manifest_text: UTF-8 JSON text of ``manifest.json``.

    Returns:
        Validated fixture set with rows in stream order.

    Raises:
        ValueError: When the manifest shape, licence, families, gold placement,
            gold provenance or per-family row counts are wrong.
    """
    try:
        raw = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        msg = "PSAI vision manifest is not valid JSON"
        raise ValueError(msg) from exc
    raw = _as_object(raw, "manifest")
    _require_keys(raw, _MANIFEST_KEYS, "manifest")
    version = int(raw["manifest_version"])
    if version != MANIFEST_VERSION:
        msg = (
            f"PSAI vision manifest_version {version} is not the supported "
            f"version {MANIFEST_VERSION}"
        )
        raise ValueError(msg)
    if str(raw["license"]) != DATASET_LICENSE:
        msg = (
            f"PSAI vision manifest license {raw['license']!r} must be "
            f"{DATASET_LICENSE!r}"
        )
        raise ValueError(msg)
    if str(raw["dataset_id"]) != DATASET_ID:
        msg = f"PSAI vision manifest dataset_id must be {DATASET_ID!r}"
        raise ValueError(msg)
    rows = _as_array(raw["rows"], "manifest rows")
    examples = tuple(_parse_row(row) for row in rows)
    _validate_rows(examples)
    return VisionSmokeFixture(
        dataset_id=str(raw["dataset_id"]),
        license=str(raw["license"]),
        split=str(raw["split"]) or SPLIT,
        source=str(raw["source"]),
        screenshot_index=int(raw["screenshot_index"]),
        gold_provenance=_parse_gold_provenance(raw["gold_provenance"]),
        examples=examples,
    )


def example_image_input(
    example: PsaiVisionExample,
    read_image: Callable[[str], bytes],
) -> ImageInput:
    """Build an ``ImageInput`` and verify the bytes against the manifest digest.

    File IO stays with the caller, so the loader runs offline in any harness.

    Args:
        example: Row whose screenshot to attach.
        read_image: Reader taking ``screenshot.file_name`` and returning bytes.

    Returns:
        Image ready for the keyword-only ``media`` argument.

    Raises:
        ScoringValidationError: When the bytes do not match the recorded
            ``sha256``, so a swapped or corrupt fixture cannot reach a model.
    """
    data = read_image(example.screenshot.file_name)
    digest = hashlib.sha256(data).hexdigest()
    if digest != example.screenshot.sha256:
        msg = (
            f"PSAI vision screenshot {example.screenshot.file_name!r} sha256 "
            f"{digest} does not match the manifest {example.screenshot.sha256}"
        )
        raise ScoringValidationError(msg)
    return ImageInput(data=data, mime_type=example.screenshot.mime_type)
