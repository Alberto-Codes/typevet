"""Offline tests for the LFW View 2 pairs loader (#300).

The pairs fixture holds synthetic names and image numbers only. Archive tests
build a small ``.tgz`` of synthetic solid-colour PNGs in ``tmp_path``. No test
reads the network or a real LFW file.
"""

from __future__ import annotations

import hashlib
import io
import struct
import tarfile
import zlib
from pathlib import Path

import httpx
import pytest

from typevet_evals.datasets.lfw import (
    ARCHIVE_FILE_NAME,
    ARCHIVE_ROOT,
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    CACHE_ENV_VAR,
    DEFAULT_PER_CLASS,
    DEFAULT_SEED,
    PAIRS_FILE_NAME,
    PAIRS_SHA256,
    PAIRS_URL,
    LfwChecksumError,
    LfwFace,
    LfwPair,
    fetch_lfw_files,
    fetch_verified,
    load_pairs,
    parse_pairs,
    read_members,
    resolve_cache_dir,
    select_balanced_slice,
)

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "lfw" / "pairs_excerpt.txt"
FIXTURE_TEXT = FIXTURE.read_text(encoding="utf-8")


def _solid_png(rgb: tuple[int, int, int], size: int = 2) -> bytes:
    """Return a solid-colour RGB PNG of ``size`` by ``size`` pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serving(payload: bytes, seen: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_source_constants_pin_the_figshare_files() -> None:
    assert PAIRS_URL == "https://ndownloader.figshare.com/files/5976006"
    assert ARCHIVE_URL == "https://ndownloader.figshare.com/files/5976015"
    assert PAIRS_SHA256 == (
        "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
    )
    assert ARCHIVE_SHA256 == (
        "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
    )
    assert (PAIRS_FILE_NAME, ARCHIVE_FILE_NAME) == ("pairs.txt", "lfw-funneled.tgz")
    assert (DEFAULT_PER_CLASS, DEFAULT_SEED) == (100, 0)


def test_parse_pairs_reads_folds_ids_and_labels() -> None:
    pairs = parse_pairs(FIXTURE_TEXT)
    assert len(pairs) == 12
    assert pairs[0] == LfwPair(
        fold=1,
        left=LfwFace("Alpha_Example", 1),
        right=LfwFace("Alpha_Example", 2),
        same_person=True,
    )
    assert pairs[3] == LfwPair(
        fold=1,
        left=LfwFace("Alpha_Example", 1),
        right=LfwFace("Delta_Example", 1),
        same_person=False,
    )
    assert [p.fold for p in pairs] == [1] * 6 + [2] * 6
    assert [p.same_person for p in pairs] == ([True] * 3 + [False] * 3) * 2


def test_face_member_path_matches_the_archive_layout() -> None:
    face = LfwFace("Alpha_Example", 7)
    assert ARCHIVE_ROOT == "lfw_funneled"
    assert face.member_path == "lfw_funneled/Alpha_Example/Alpha_Example_0007.jpg"
    pair = parse_pairs(FIXTURE_TEXT)[3]
    assert pair.member_paths == (
        "lfw_funneled/Alpha_Example/Alpha_Example_0001.jpg",
        "lfw_funneled/Delta_Example/Delta_Example_0001.jpg",
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "header"),
        ("10\n", "header"),
        ("x\t3\n", "header"),
        ("0\t3\n", "header"),
        ("1\t1\nA\t1\t2\n", "expected 2"),
        ("1\t1\nA\t1\t2\nB\t1\t2\n", "mismatch line"),
        ("1\t1\nA\t1\tB\t2\nB\t1\tC\t2\n", "match line"),
        ("1\t1\nA\tx\t2\nB\t1\tC\t2\n", "image number"),
        ("1\t1\nA\t0\t2\nB\t1\tC\t2\n", "image number"),
    ],
)
def test_parse_pairs_rejects_malformed_text(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_pairs(text)


def test_balanced_slice_is_deterministic_balanced_and_spread() -> None:
    pairs = parse_pairs(FIXTURE_TEXT)
    chosen = select_balanced_slice(pairs, per_class=2, seed=0)
    assert chosen == select_balanced_slice(pairs, per_class=2, seed=0)
    assert sum(p.same_person for p in chosen) == 2
    assert sum(not p.same_person for p in chosen) == 2
    assert {p.fold for p in chosen if p.same_person} == {1, 2}
    assert {p.fold for p in chosen if not p.same_person} == {1, 2}
    assert len(set(chosen)) == 4
    assert all(p in pairs for p in chosen)


def _full_view2_text(folds: int = 10, per_class: int = 300) -> str:
    """Return a synthetic ``pairs.txt`` with the full View 2 shape."""
    lines = [f"{folds}\t{per_class}"]
    for fold in range(1, folds + 1):
        lines += [f"Same_F{fold}_P{k}\t1\t2" for k in range(per_class)]
        lines += [
            f"Left_F{fold}_P{k}\t1\tRight_F{fold}_P{k}\t1" for k in range(per_class)
        ]
    return "\n".join(lines) + "\n"


def test_default_slice_takes_ten_of_each_class_from_every_fold() -> None:
    pairs = parse_pairs(_full_view2_text())
    assert len(pairs) == 6000
    chosen = select_balanced_slice(pairs)
    assert len(chosen) == 200
    assert len(set(chosen)) == 200
    for fold in range(1, 11):
        in_fold = [p for p in chosen if p.fold == fold]
        assert sum(p.same_person for p in in_fold) == 10
        assert sum(not p.same_person for p in in_fold) == 10


def test_balanced_slice_takes_every_pair_when_asked() -> None:
    pairs = parse_pairs(FIXTURE_TEXT)
    chosen = select_balanced_slice(pairs, per_class=6)
    assert sorted(chosen, key=pairs.index) == list(pairs)


def test_balanced_slice_seed_changes_the_selection_order() -> None:
    pairs = parse_pairs(FIXTURE_TEXT)
    orders = {select_balanced_slice(pairs, per_class=3, seed=s) for s in range(8)}
    assert len(orders) > 1


@pytest.mark.parametrize("per_class", [0, 7])
def test_balanced_slice_rejects_impossible_sizes(per_class: int) -> None:
    with pytest.raises(ValueError, match="per_class"):
        select_balanced_slice(parse_pairs(FIXTURE_TEXT), per_class=per_class)


def test_cache_dir_argument_wins_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))
    assert resolve_cache_dir(tmp_path / "arg") == tmp_path / "arg"
    assert resolve_cache_dir() == tmp_path / "env"


def test_cache_dir_default_is_under_the_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_cache_dir() == tmp_path / ".cache" / "typevet" / "lfw"


def test_fetch_verified_writes_once_then_reuses_the_cache(tmp_path: Path) -> None:
    payload = b"synthetic pairs payload"
    seen: list[str] = []
    dest = tmp_path / "cache" / "pairs.txt"
    with _serving(payload, seen) as client:
        path = fetch_verified(PAIRS_URL, _sha256(payload), dest, client=client)
        again = fetch_verified(PAIRS_URL, _sha256(payload), dest, client=client)
    assert path == again == dest
    assert dest.read_bytes() == payload
    assert seen == [PAIRS_URL]


def test_fetch_verified_refuses_a_download_with_another_hash(tmp_path: Path) -> None:
    dest = tmp_path / "pairs.txt"
    with (
        _serving(b"tampered", []) as client,
        pytest.raises(LfwChecksumError, match="SHA-256"),
    ):
        fetch_verified(PAIRS_URL, _sha256(b"expected"), dest, client=client)
    assert list(tmp_path.iterdir()) == []


def test_fetch_verified_refuses_a_cached_file_with_another_hash(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "pairs.txt"
    dest.write_bytes(b"stale")
    seen: list[str] = []
    with (
        _serving(b"expected", seen) as client,
        pytest.raises(LfwChecksumError, match="SHA-256"),
    ):
        fetch_verified(PAIRS_URL, _sha256(b"expected"), dest, client=client)
    assert seen == []
    assert dest.read_bytes() == b"stale"


def test_fetch_lfw_files_refuses_bytes_that_are_not_the_pinned_files(
    tmp_path: Path,
) -> None:
    seen: list[str] = []
    with (
        _serving(FIXTURE_TEXT.encode(), seen) as client,
        pytest.raises(LfwChecksumError),
    ):
        fetch_lfw_files(cache_dir=tmp_path, client=client)
    assert seen == [PAIRS_URL]
    assert list(tmp_path.iterdir()) == []


def test_fetch_lfw_files_returns_cached_paths_that_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_bytes = FIXTURE_TEXT.encode()
    archive_bytes = b"synthetic archive"
    (tmp_path / PAIRS_FILE_NAME).write_bytes(pairs_bytes)
    (tmp_path / ARCHIVE_FILE_NAME).write_bytes(archive_bytes)
    monkeypatch.setattr("typevet_evals.datasets.lfw.PAIRS_SHA256", _sha256(pairs_bytes))
    monkeypatch.setattr(
        "typevet_evals.datasets.lfw.ARCHIVE_SHA256", _sha256(archive_bytes)
    )
    seen: list[str] = []
    with _serving(b"", seen) as client:
        files = fetch_lfw_files(cache_dir=tmp_path, client=client)
    assert files.pairs_path == tmp_path / PAIRS_FILE_NAME
    assert files.archive_path == tmp_path / ARCHIVE_FILE_NAME
    assert seen == []
    assert load_pairs(files.pairs_path) == parse_pairs(FIXTURE_TEXT)


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_read_members_returns_requested_images_by_member_path(
    tmp_path: Path,
) -> None:
    red, blue = _solid_png((255, 0, 0)), _solid_png((0, 0, 255))
    pair = parse_pairs(FIXTURE_TEXT)[3]
    left, right = pair.member_paths
    decoy = "other/Alpha_Example/Alpha_Example_0001.jpg"
    archive = tmp_path / "faces.tgz"
    _write_archive(
        archive,
        {decoy: b"decoy", left: red, "lfw_funneled/Other/x.jpg": b"x", right: blue},
    )
    images = read_members(archive, [right, left])
    assert images == {left: red, right: blue}


def test_read_members_refuses_a_link_at_a_requested_path(tmp_path: Path) -> None:
    target = "lfw_funneled/A/A_0001.jpg"
    link = "lfw_funneled/B/B_0001.jpg"
    archive = tmp_path / "faces.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(target)
        info.size = 1
        handle.addfile(info, io.BytesIO(b"a"))
        info = tarfile.TarInfo(link)
        info.type = tarfile.SYMTYPE
        info.linkname = "../A/A_0001.jpg"
        handle.addfile(info)
    with pytest.raises(KeyError, match="B_0001"):
        read_members(archive, [link])


def test_read_members_reports_a_missing_member(tmp_path: Path) -> None:
    archive = tmp_path / "faces.tgz"
    _write_archive(archive, {"lfw_funneled/A/A_0001.jpg": b"a"})
    with pytest.raises(KeyError, match="B_0001"):
        read_members(archive, ["lfw_funneled/B/B_0001.jpg"])
