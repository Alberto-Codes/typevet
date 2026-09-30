"""CEDAR signature pairs loader for the signature-match evaluation (#318).

The CEDAR offline signature data set holds 55 writers. Each writer has 24
genuine signatures and 24 skilled forgeries, as PNG images in one RAR
archive. This module names every image by writer, sample and kind, picks a
deterministic balanced slice of pairs and reads images from the archive by
member path.

The loader fetches the archive from CEDAR into a cache outside the
repository and refuses a file whose SHA-256 differs from the pinned value.
CEDAR lists the archive under "Published Data Sets" on its publications page
without a sign-in step. The page states no licence. The data set is used for
research only. No signature bytes go into the repository.

Archive layout, read from the pinned archive on 2026-09-30:
``signatures/full_org/original_<writer>_<sample>.png`` for genuine
signatures, ``signatures/full_forg/forgeries_<writer>_<sample>.png`` for
skilled forgeries, writers 1 to 55 and samples 1 to 24 without zero padding,
plus ``Readme.txt`` and two ``Thumbs.db`` files.

Attributes:
    ARCHIVE_URL (str): CEDAR download URL for ``signatures.rar``.
    SOURCE_PAGE_URL (str): CEDAR page that links the archive.
    ARCHIVE_SHA256 (str): Pinned SHA-256 of ``signatures.rar``.
    ARCHIVE_FILE_NAME (str): Cache file name for the archive.
    CACHE_ENV_VAR (str): Environment variable that sets the cache directory.
    UNRAR_COMMAND (str): Command name of the RAR extraction tool.
    UNRAR_NO_FILES_STATUS (int): ``unrar`` exit status when no requested
        member is in the archive.
    WRITER_COUNT (int): Writers in the data set.
    SAMPLES_PER_WRITER (int): Genuine signatures, and forgeries, per writer.
    DEFAULT_PER_KIND (int): Pairs per pair kind in the default slice.
    DEFAULT_SEED (int): Default slice seed.

Examples:
    ```python
    from typevet_evals.datasets.cedar import (
        fetch_cedar_archive,
        read_members,
        select_balanced_slice,
    )

    archive = fetch_cedar_archive()
    pairs = select_balanced_slice()
    paths = [path for pair in pairs for path in pair.member_paths]
    images = read_members(archive, paths)
    ```

See Also:
    - [typevet_evals.signature_match][]: one judgment request per pair
    - [typevet_evals.datasets.lfw][]: the verified download helper
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import httpx

from typevet_evals.datasets.lfw import fetch_verified

ARCHIVE_URL: Final[str] = "https://cedar.buffalo.edu/NIJ/data/signatures.rar"
SOURCE_PAGE_URL: Final[str] = "https://cedar.buffalo.edu/NIJ/publications.html"
ARCHIVE_SHA256: Final[str] = (
    "f74b859352783b82399c1be48078b79ad637160ba11f16baf92911dd5568f4d6"
)
ARCHIVE_FILE_NAME: Final[str] = "signatures.rar"
CACHE_ENV_VAR: Final[str] = "TYPEVET_CEDAR_CACHE"
UNRAR_COMMAND: Final[str] = "unrar"
UNRAR_NO_FILES_STATUS: Final[int] = 10
WRITER_COUNT: Final[int] = 55
SAMPLES_PER_WRITER: Final[int] = 24
DEFAULT_PER_KIND: Final[int] = 60
DEFAULT_SEED: Final[int] = 0

_ROOT: Final[str] = "signatures"
_GENUINE: Final[tuple[str, str]] = ("full_org", "original")
_FORGED: Final[tuple[str, str]] = ("full_forg", "forgeries")
_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"signatures/(full_org|full_forg)/(original|forgeries)_(\d+)_(\d+)\.png"
)

Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class CedarToolError(RuntimeError):
    """The RAR extraction tool is not installed.

    Examples:
        ```python
        try:
            read_members(archive, paths)
        except CedarToolError:
            ...
        ```
    """


class PairKind(StrEnum):
    """How image 2 of a pair relates to the genuine image 1.

    Attributes:
        GENUINE_GENUINE (str): Another genuine signature by the same writer.
        GENUINE_SKILLED (str): A skilled forgery of the same writer's signature.
        GENUINE_RANDOM (str): A genuine signature by another writer.

    Examples:
        ```python
        assert PairKind.GENUINE_SKILLED == "genuine_skilled"
        ```
    """

    GENUINE_GENUINE = "genuine_genuine"
    GENUINE_SKILLED = "genuine_skilled"
    GENUINE_RANDOM = "genuine_random"


@dataclass(frozen=True, slots=True)
class CedarSignature:
    """One CEDAR image, named by writer, sample number and kind.

    Attributes:
        writer (int): One-based writer number.
        sample (int): One-based sample number for that writer and kind.
        forged (bool): ``True`` for a skilled forgery.

    Examples:
        ```python
        CedarSignature(10, 1, forged=False).member_path
        ```
    """

    writer: int
    sample: int
    forged: bool

    @property
    def signature_id(self) -> str:
        """Return the file stem, for example ``original_10_1``.

        Returns:
            ``<original|forgeries>_<writer>_<sample>``.
        """
        prefix = _FORGED[1] if self.forged else _GENUINE[1]
        return f"{prefix}_{self.writer}_{self.sample}"

    @property
    def member_path(self) -> str:
        """Return the archive member path of this image.

        Returns:
            ``signatures/<full_org|full_forg>/<signature_id>.png``.
        """
        folder = _FORGED[0] if self.forged else _GENUINE[0]
        return f"{_ROOT}/{folder}/{self.signature_id}.png"


@dataclass(frozen=True, slots=True)
class CedarPair:
    """One pair: a genuine image 1 and a questioned image 2.

    Attributes:
        kind (PairKind): Gold relation of image 2 to image 1.
        reference (CedarSignature): Image 1, always genuine.
        questioned (CedarSignature): Image 2.

    Examples:
        ```python
        pair = CedarPair(
            PairKind.GENUINE_GENUINE,
            CedarSignature(1, 1, forged=False),
            CedarSignature(1, 2, forged=False),
        )
        assert pair.same_writer
        ```
    """

    kind: PairKind
    reference: CedarSignature
    questioned: CedarSignature

    @property
    def same_writer(self) -> bool:
        """Return the gold same-writer label.

        Returns:
            ``True`` only for a genuine-genuine pair.
        """
        return self.kind is PairKind.GENUINE_GENUINE

    @property
    def pair_id(self) -> str:
        """Return a stable id for receipts.

        Returns:
            ``<kind>:<image 1 signature id>:<image 2 signature id>``.
        """
        return (
            f"{self.kind}:{self.reference.signature_id}:{self.questioned.signature_id}"
        )

    @property
    def member_paths(self) -> tuple[str, str]:
        """Return the archive member paths of image 1 and image 2.

        Returns:
            The reference and questioned member paths, in that order.
        """
        return (self.reference.member_path, self.questioned.member_path)


def parse_member_path(member_path: str) -> CedarSignature | None:
    """Parse one archive member path into a signature.

    The CEDAR naming rules live here only. Correct this function if the
    archive layout changes.

    Args:
        member_path: Member path as the archive lists it.

    Returns:
        The signature, or ``None`` for a member that is not a signature image,
        such as a directory, ``Readme.txt`` or ``Thumbs.db``.

    Raises:
        ValueError: When a signature image name disagrees with its folder or
            holds a number below 1.
    """
    match = _MEMBER.fullmatch(member_path)
    if match is None:
        return None
    folder, prefix, writer, sample = match.groups()
    if (folder, prefix) not in (_GENUINE, _FORGED) or min(int(writer), int(sample)) < 1:
        msg = f"CEDAR member {member_path!r} does not follow the archive layout"
        raise ValueError(msg)
    return CedarSignature(int(writer), int(sample), forged=prefix == _FORGED[1])


def catalog(
    *, writers: int = WRITER_COUNT, samples: int = SAMPLES_PER_WRITER
) -> tuple[CedarSignature, ...]:
    """Return every signature id of the data set, without reading the archive.

    Args:
        writers: Writer count.
        samples: Genuine signatures, and forgeries, per writer.

    Returns:
        Each writer's genuine signatures, then its forgeries, writer by writer.
    """
    return tuple(
        CedarSignature(writer, sample, forged=forged)
        for writer in range(1, writers + 1)
        for forged in (False, True)
        for sample in range(1, samples + 1)
    )


def _key(seed: int, salt: str, text: str) -> bytes:
    return hashlib.sha256(f"{seed}:{salt}:{text}".encode()).digest()


@dataclass(frozen=True, slots=True)
class _Pool:
    seed: int
    salt: str
    groups: dict[tuple[int, bool], list[CedarSignature]]

    def ordered(self, writer: int, *, forged: bool) -> list[CedarSignature]:
        """Return one writer's images of one kind in seeded order.

        Args:
            writer: Writer number.
            forged: ``True`` for forgeries, ``False`` for genuine images.

        Returns:
            The images, ordered by SHA-256 key.
        """
        items = self.groups.get((writer, forged), [])
        return sorted(items, key=lambda s: _key(self.seed, self.salt, s.signature_id))

    def rotation(self, *, genuine: int, forged: int) -> list[int]:
        """Return the writers with enough images, in seeded order.

        Args:
            genuine: Genuine images a writer needs.
            forged: Forgeries a writer needs.

        Returns:
            Writer numbers ordered by SHA-256 key.
        """
        writers = {
            writer
            for writer, _ in self.groups
            if len(self.groups.get((writer, False), [])) >= genuine
            and len(self.groups.get((writer, True), [])) >= forged
        }
        return sorted(writers, key=lambda w: _key(self.seed, self.salt, str(w)))


_Maker = Callable[[_Pool, Sequence[int], int, int], CedarPair | None]


def _genuine_pair(
    pool: _Pool, rotation: Sequence[int], index: int, depth: int
) -> CedarPair | None:
    items = pool.ordered(rotation[index], forged=False)
    if 2 * depth + 1 >= len(items):
        return None
    return CedarPair(PairKind.GENUINE_GENUINE, items[2 * depth], items[2 * depth + 1])


def _skilled_pair(
    pool: _Pool, rotation: Sequence[int], index: int, depth: int
) -> CedarPair | None:
    genuine = pool.ordered(rotation[index], forged=False)
    forged = pool.ordered(rotation[index], forged=True)
    if depth >= min(len(genuine), len(forged)):
        return None
    return CedarPair(PairKind.GENUINE_SKILLED, genuine[depth], forged[depth])


def _random_pair(
    pool: _Pool, rotation: Sequence[int], index: int, depth: int
) -> CedarPair | None:
    size = len(rotation)
    if size <= 1:
        return None
    other = rotation[(index + 1 + depth % (size - 1)) % size]
    own = pool.ordered(rotation[index], forged=False)
    theirs = pool.ordered(other, forged=False)
    if depth >= min(len(own), len(theirs)):
        return None
    return CedarPair(PairKind.GENUINE_RANDOM, own[depth], theirs[depth])


# Pair maker, then the genuine and forged images a writer needs for the kind.
_RULES: Final[dict[PairKind, tuple[_Maker, int, int]]] = {
    PairKind.GENUINE_GENUINE: (_genuine_pair, 2, 0),
    PairKind.GENUINE_SKILLED: (_skilled_pair, 1, 1),
    PairKind.GENUINE_RANDOM: (_random_pair, 1, 0),
}


def _take(
    kind: PairKind,
    groups: dict[tuple[int, bool], list[CedarSignature]],
    count: int,
    seed: int,
) -> list[CedarPair]:
    make, genuine, forged = _RULES[kind]
    pool = _Pool(seed, str(kind), groups)
    rotation = pool.rotation(genuine=genuine, forged=forged)
    depth_limit = max((len(items) for items in groups.values()), default=0)
    chosen: list[CedarPair] = []
    for depth in range(depth_limit):
        for index in range(len(rotation)):
            pair = make(pool, rotation, index, depth) if len(chosen) < count else None
            if pair is not None:
                chosen.append(pair)
    if len(chosen) < count:
        msg = f"only {len(chosen)} {kind} pairs available; per_kind is {count}"
        raise ValueError(msg)
    return chosen


def select_balanced_slice(
    signatures: Iterable[CedarSignature] | None = None,
    *,
    per_kind: int = DEFAULT_PER_KIND,
    seed: int = DEFAULT_SEED,
) -> tuple[CedarPair, ...]:
    """Pick a deterministic slice with equal counts of each pair kind.

    Each kind visits the writers in a seeded order, one pair per writer per
    round, so the slice spreads over every writer before a writer repeats.
    Image 1 is always genuine. A random pair takes image 2 from another
    writer that the round shifts. Pairs never repeat. The same inputs always
    give the same slice in the same order.

    Args:
        signatures: Available signatures; defaults to :func:`catalog`.
        per_kind: Pairs to take for each :class:`PairKind`.
        seed: Seed for the SHA-256 keys that order writers, samples and the
            result.

    Returns:
        ``3 * per_kind`` pairs with the kinds mixed.

    Raises:
        ValueError: When ``per_kind`` is below 1 or a kind has too few pairs.
    """
    if per_kind < 1:
        msg = f"per_kind must be at least 1, got {per_kind}"
        raise ValueError(msg)
    groups: dict[tuple[int, bool], list[CedarSignature]] = {}
    for item in catalog() if signatures is None else signatures:
        groups.setdefault((item.writer, item.forged), []).append(item)
    chosen = [pair for kind in PairKind for pair in _take(kind, groups, per_kind, seed)]
    return tuple(sorted(chosen, key=lambda pair: _key(seed, "slice", pair.pair_id)))


def resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the CEDAR cache directory.

    Args:
        cache_dir: Explicit directory; wins over the environment.

    Returns:
        ``cache_dir``, else ``$TYPEVET_CEDAR_CACHE``, else
        ``~/.cache/typevet/cedar``.
    """
    if cache_dir is not None:
        return cache_dir
    configured = os.environ.get(CACHE_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "typevet" / "cedar"


def fetch_cedar_archive(
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Fetch and verify ``signatures.rar``.

    A cached file is checked and never replaced. A download goes to a
    temporary file and moves into the cache only when its hash matches.

    Args:
        cache_dir: Cache directory; see :func:`resolve_cache_dir`.
        client: HTTP client; inject one for tests.

    Returns:
        Local path of the verified archive.

    Raises:
        typevet_evals.datasets.lfw.LfwChecksumError: When the cached or
            downloaded file does not match :data:`ARCHIVE_SHA256`. The error
            comes from the shared download helper.
        httpx.HTTPError: When the download fails.
    """
    dest = resolve_cache_dir(cache_dir) / ARCHIVE_FILE_NAME
    return fetch_verified(ARCHIVE_URL, ARCHIVE_SHA256, dest, client=client)


def read_members(
    archive_path: Path,
    member_paths: Sequence[str],
    *,
    unrar: str | None = None,
    run: Runner = subprocess.run,
) -> dict[str, bytes]:
    """Read image bytes from the RAR archive by member path.

    Every requested name must be a signature member path that
    :func:`parse_member_path` accepts, so no name can point outside the
    temporary directory. The ``unrar`` tool extracts the requested members
    once into that directory. The directory is removed before the function
    returns. ``unrar`` exit status 10 means that no requested member is in
    the archive; the function reports it as missing members.

    Args:
        archive_path: Local ``signatures.rar`` path.
        member_paths: Member paths, for example from ``CedarPair.member_paths``.
        unrar: Path of the ``unrar`` tool; found on ``PATH`` when omitted.
        run: Command runner; inject one for tests.

    Returns:
        Member path to file bytes, one entry per distinct requested path.

    Raises:
        ValueError: When a requested name is not a signature member path.
        FileNotFoundError: When ``archive_path`` does not exist.
        CedarToolError: When no ``unrar`` tool is found.
        KeyError: When a requested member is not in the archive.
        subprocess.CalledProcessError: When the tool fails with another exit
            status.
    """
    wanted = sorted(set(member_paths))
    for name in wanted:
        if parse_member_path(name) is None:
            msg = f"{name!r} is not a CEDAR signature member path"
            raise ValueError(msg)
    tool = unrar or shutil.which(UNRAR_COMMAND)
    if tool is None:
        msg = (
            f"{UNRAR_COMMAND} not found on PATH; install it to read {ARCHIVE_FILE_NAME}"
        )
        raise CedarToolError(msg)
    if not archive_path.is_file():
        msg = f"CEDAR archive not found: {archive_path}"
        raise FileNotFoundError(msg)
    with tempfile.TemporaryDirectory(prefix="typevet-cedar-") as scratch:
        root = Path(scratch)
        list_file = root / "members.txt"
        list_file.write_text("\n".join(wanted) + "\n", encoding="utf-8")
        dest = root / "out"
        dest.mkdir()
        command = [tool, "x", "-inul", "-y", str(archive_path)]
        command += [f"@{list_file}", f"{dest}{os.sep}"]
        result = run(command, check=False, capture_output=True)
        if result.returncode not in (0, UNRAR_NO_FILES_STATUS):
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )
        found = {
            name: (dest / name).read_bytes()
            for name in wanted
            if (dest / name).is_file()
        }
    missing = [name for name in wanted if name not in found]
    if missing:
        msg = f"archive members not found: {', '.join(missing)}"
        raise KeyError(msg)
    return found
