"""LFW View 2 pairs loader for the face-match evaluation (#300).

Labeled Faces in the Wild (LFW) View 2 lists 6,000 face pairs in ten folds.
Each fold holds 300 same-person pairs, then 300 different-person pairs.
This module parses ``pairs.txt``, picks a deterministic balanced slice and
reads face images from the funneled archive by member path.

The loader fetches both files from figshare into a cache outside the
repository and refuses any file whose SHA-256 differs from the pinned value.
LFW has no formal licence. The photographers keep the image copyright. The
dataset is for research use. No face bytes go into the repository.

Attributes:
    PAIRS_URL (str): figshare download URL for ``pairs.txt``.
    ARCHIVE_URL (str): figshare download URL for ``lfw-funneled.tgz``.
    PAIRS_SHA256 (str): Pinned SHA-256 of ``pairs.txt``.
    ARCHIVE_SHA256 (str): Pinned SHA-256 of ``lfw-funneled.tgz``.
    PAIRS_FILE_NAME (str): Cache file name for the pairs list.
    ARCHIVE_FILE_NAME (str): Cache file name for the image archive.
    ARCHIVE_ROOT (str): Top directory of every image member in the archive.
    CACHE_ENV_VAR (str): Environment variable that sets the cache directory.
    DEFAULT_PER_CLASS (int): Pairs per class in the default slice.
    DEFAULT_SEED (int): Default slice seed.

Examples:
    ```python
    from typevet_evals.datasets.lfw import (
        fetch_lfw_files,
        load_pairs,
        read_members,
        select_balanced_slice,
    )

    files = fetch_lfw_files()
    pairs = select_balanced_slice(load_pairs(files.pairs_path))
    paths = [path for pair in pairs for path in pair.member_paths]
    images = read_members(files.archive_path, paths)
    ```

See Also:
    - [typevet_evals.face_match][]: one judgment request per pair
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

PAIRS_URL: Final[str] = "https://ndownloader.figshare.com/files/5976006"
ARCHIVE_URL: Final[str] = "https://ndownloader.figshare.com/files/5976015"
PAIRS_SHA256: Final[str] = (
    "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
)
ARCHIVE_SHA256: Final[str] = (
    "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
)
PAIRS_FILE_NAME: Final[str] = "pairs.txt"
ARCHIVE_FILE_NAME: Final[str] = "lfw-funneled.tgz"
ARCHIVE_ROOT: Final[str] = "lfw_funneled"
CACHE_ENV_VAR: Final[str] = "TYPEVET_LFW_CACHE"
DEFAULT_PER_CLASS: Final[int] = 100
DEFAULT_SEED: Final[int] = 0

_HEADER_FIELDS: Final[int] = 2
_MATCH_FIELDS: Final[int] = 3
_MISMATCH_FIELDS: Final[int] = 4
_BLOCK_SIZE: Final[int] = 1 << 20
_DOWNLOAD_TIMEOUT: Final[float] = 600.0


class LfwChecksumError(ValueError):
    """A downloaded or cached LFW file has an unexpected SHA-256.

    Examples:
        ```python
        try:
            fetch_lfw_files()
        except LfwChecksumError:
            ...
        ```
    """


@dataclass(frozen=True, slots=True)
class LfwFace:
    """One LFW image, named by person and image number.

    Attributes:
        name (str): Person directory name, with underscores for spaces.
        number (int): One-based image number for that person.

    Examples:
        ```python
        LfwFace("Some_Person", 1).member_path
        ```
    """

    name: str
    number: int

    @property
    def member_path(self) -> str:
        """Return the archive member path of this image.

        Returns:
            ``lfw_funneled/<name>/<name>_<number:04d>.jpg``.
        """
        return f"{ARCHIVE_ROOT}/{self.name}/{self.name}_{self.number:04d}.jpg"


@dataclass(frozen=True, slots=True)
class LfwPair:
    """One View 2 pair with its fold and gold label.

    Attributes:
        fold (int): One-based fold number.
        left (LfwFace): Image 1 of the pair.
        right (LfwFace): Image 2 of the pair.
        same_person (bool): Gold label; ``True`` for a same-person line.

    Examples:
        ```python
        pair = LfwPair(1, LfwFace("A", 1), LfwFace("A", 2), same_person=True)
        left_path, right_path = pair.member_paths
        ```
    """

    fold: int
    left: LfwFace
    right: LfwFace
    same_person: bool

    @property
    def member_paths(self) -> tuple[str, str]:
        """Return the archive member paths of image 1 and image 2.

        Returns:
            The left and right member paths, in that order.
        """
        return (self.left.member_path, self.right.member_path)


@dataclass(frozen=True, slots=True)
class LfwFiles:
    """Verified local paths of the two LFW files.

    Attributes:
        pairs_path (Path): Cached ``pairs.txt``.
        archive_path (Path): Cached ``lfw-funneled.tgz``.

    Examples:
        ```python
        files = fetch_lfw_files()
        pairs = load_pairs(files.pairs_path)
        ```
    """

    pairs_path: Path
    archive_path: Path


def _positive_int(text: str, what: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"{what} must be a positive integer, got {text!r}"
        raise ValueError(msg)
    return value


def _parse_header(line: str) -> tuple[int, int]:
    fields = line.split("\t")
    if len(fields) != _HEADER_FIELDS:
        msg = f"malformed pairs header {line!r}; expected two tab-separated counts"
        raise ValueError(msg)
    return (
        _positive_int(fields[0], "pairs header fold count"),
        _positive_int(fields[1], "pairs header per-fold count"),
    )


def _parse_line(line: str, *, fold: int, same_person: bool) -> LfwPair:
    fields = line.split("\t")
    if same_person:
        if len(fields) != _MATCH_FIELDS:
            msg = f"fold {fold}: expected a match line (name, i, j), got {line!r}"
            raise ValueError(msg)
        name, left, right = fields
        return LfwPair(
            fold,
            LfwFace(name, _positive_int(left, "image number")),
            LfwFace(name, _positive_int(right, "image number")),
            same_person=True,
        )
    if len(fields) != _MISMATCH_FIELDS:
        msg = (
            f"fold {fold}: expected a mismatch line (name1, i, name2, j), got {line!r}"
        )
        raise ValueError(msg)
    left_name, left, right_name, right = fields
    return LfwPair(
        fold,
        LfwFace(left_name, _positive_int(left, "image number")),
        LfwFace(right_name, _positive_int(right, "image number")),
        same_person=False,
    )


def parse_pairs(text: str) -> tuple[LfwPair, ...]:
    """Parse View 2 ``pairs.txt`` text into pairs in file order.

    Args:
        text: File text. The first line holds the fold count and the pairs per
            class per fold, separated by a tab. Each fold then holds that many
            match lines, then that many mismatch lines.

    Returns:
        Every pair with its one-based fold and gold label.

    Raises:
        ValueError: When the header, the line count or a line is malformed.
    """
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        msg = "empty pairs text; expected a header line"
        raise ValueError(msg)
    folds, per_fold = _parse_header(lines[0])
    body = lines[1:]
    expected = folds * per_fold * 2
    if len(body) != expected:
        msg = f"expected {expected} pair lines after the header, found {len(body)}"
        raise ValueError(msg)
    pairs: list[LfwPair] = []
    for index, line in enumerate(body):
        fold, offset = divmod(index, per_fold * 2)
        pairs.append(_parse_line(line, fold=fold + 1, same_person=offset < per_fold))
    return tuple(pairs)


def load_pairs(path: Path) -> tuple[LfwPair, ...]:
    """Read and parse a local ``pairs.txt``.

    Args:
        path: File path, usually from :func:`fetch_lfw_files`.

    Returns:
        Every pair in file order.
    """
    return parse_pairs(path.read_text(encoding="utf-8"))


def _seeded_key(seed: int, salt: str, pair: LfwPair) -> bytes:
    left, right = pair.member_paths
    return hashlib.sha256(f"{seed}:{salt}:{pair.fold}:{left}:{right}".encode()).digest()


def _spread(pairs: list[LfwPair], count: int, seed: int) -> list[LfwPair]:
    by_fold: dict[int, list[LfwPair]] = {}
    for pair in pairs:
        by_fold.setdefault(pair.fold, []).append(pair)
    queues = [
        sorted(by_fold[fold], key=lambda pair: _seeded_key(seed, "fold", pair))
        for fold in sorted(by_fold)
    ]
    chosen: list[LfwPair] = []
    depth = 0
    while len(chosen) < count:
        for queue in queues:
            if depth < len(queue) and len(chosen) < count:
                chosen.append(queue[depth])
        depth += 1
    return chosen


def select_balanced_slice(
    pairs: Iterable[LfwPair],
    *,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> tuple[LfwPair, ...]:
    """Pick a deterministic slice with equal same and different pairs.

    Each class takes pairs from the folds in turn, so the slice spreads over
    every fold. The same ``pairs``, ``per_class`` and ``seed`` always give the
    same slice in the same order.

    Args:
        pairs: Parsed pairs, for example from :func:`parse_pairs`.
        per_class: Pairs to take from each class.
        seed: Seed for the SHA-256 keys that order each fold and the result.

    Returns:
        ``2 * per_class`` pairs with both classes mixed.

    Raises:
        ValueError: When ``per_class`` is below 1 or above a class size.
    """
    same = [pair for pair in pairs if pair.same_person]
    different = [pair for pair in pairs if not pair.same_person]
    limit = min(len(same), len(different))
    if per_class < 1 or per_class > limit:
        msg = f"per_class must be between 1 and {limit}, got {per_class}"
        raise ValueError(msg)
    chosen = _spread(same, per_class, seed) + _spread(different, per_class, seed)
    return tuple(sorted(chosen, key=lambda pair: _seeded_key(seed, "slice", pair)))


def resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the LFW cache directory.

    Args:
        cache_dir: Explicit directory; wins over the environment.

    Returns:
        ``cache_dir``, else ``$TYPEVET_LFW_CACHE``, else
        ``~/.cache/typevet/lfw``.
    """
    if cache_dir is not None:
        return cache_dir
    configured = os.environ.get(CACHE_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "typevet" / "lfw"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, expected_sha256: str, dest: Path, client: httpx.Client) -> None:
    partial = dest.with_name(f"{dest.name}.part")
    digest = hashlib.sha256()
    try:
        with client.stream(
            "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_bytes():
                    digest.update(block)
                    handle.write(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            msg = (
                f"refusing {url}: SHA-256 {actual} does not match "
                f"pinned {expected_sha256}"
            )
            raise LfwChecksumError(msg)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def fetch_verified(
    url: str,
    expected_sha256: str,
    dest: Path,
    *,
    client: httpx.Client | None = None,
) -> Path:
    """Return ``dest`` after a SHA-256 check, downloading it when absent.

    A cached file is checked and never replaced. A download goes to a
    temporary file and moves to ``dest`` only when its hash matches.

    Args:
        url: Source URL.
        expected_sha256: Pinned lowercase hex SHA-256.
        dest: Cache file path.
        client: HTTP client; a temporary client is opened when omitted.

    Returns:
        ``dest``, holding bytes whose SHA-256 is ``expected_sha256``.

    Raises:
        LfwChecksumError: When the cached or downloaded bytes have another
            SHA-256. A mismatched download leaves no file behind.
        httpx.HTTPError: When the download fails.
    """
    if dest.exists():
        actual = _file_sha256(dest)
        if actual != expected_sha256:
            msg = (
                f"refusing cached {dest}: SHA-256 {actual} does not match "
                f"pinned {expected_sha256}; remove the file to fetch it again"
            )
            raise LfwChecksumError(msg)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        with httpx.Client() as owned:
            _download(url, expected_sha256, dest, owned)
    else:
        _download(url, expected_sha256, dest, client)
    return dest


def fetch_lfw_files(
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> LfwFiles:
    """Fetch and verify ``pairs.txt`` and the funneled image archive.

    Args:
        cache_dir: Cache directory; see :func:`resolve_cache_dir`.
        client: HTTP client; inject one for tests.

    Returns:
        Local paths of both verified files.

    Raises:
        LfwChecksumError: When a file does not match its pinned SHA-256.
        httpx.HTTPError: When a download fails.
    """
    root = resolve_cache_dir(cache_dir)
    pairs_path = fetch_verified(
        PAIRS_URL, PAIRS_SHA256, root / PAIRS_FILE_NAME, client=client
    )
    archive_path = fetch_verified(
        ARCHIVE_URL, ARCHIVE_SHA256, root / ARCHIVE_FILE_NAME, client=client
    )
    return LfwFiles(pairs_path=pairs_path, archive_path=archive_path)


def read_members(archive_path: Path, member_paths: Iterable[str]) -> dict[str, bytes]:
    """Read image bytes from a gzip tar archive by member path.

    The archive is read once, in stream order. Nothing is extracted to disk.

    Args:
        archive_path: Local ``.tgz`` path.
        member_paths: Member paths, for example from ``LfwPair.member_paths``.

    Returns:
        Member path to file bytes, one entry per distinct requested path.

    Raises:
        KeyError: When a requested member is not a file in the archive.
    """
    wanted = set(member_paths)
    found: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                found[member.name] = handle.read()
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - found.keys())
    if missing:
        msg = f"archive members not found: {', '.join(missing)}"
        raise KeyError(msg)
    return found
