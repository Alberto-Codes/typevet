"""Offline tests for the CEDAR signature pair loader (#318).

The member fixture holds archive member names only. Download tests inject an
HTTP client and a temporary cache. Archive reader tests inject a fake
extraction command that writes synthetic solid-colour PNGs, so no test needs
the network, a RAR tool or a real CEDAR file. The repository holds no
signature bytes.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import zlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from typevet_evals.datasets import cedar
from typevet_evals.datasets.cedar import (
    ARCHIVE_FILE_NAME,
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    CACHE_ENV_VAR,
    DEFAULT_PER_KIND,
    DEFAULT_SEED,
    SAMPLES_PER_WRITER,
    SOURCE_PAGE_URL,
    UNRAR_COMMAND,
    WRITER_COUNT,
    CedarPair,
    CedarSignature,
    CedarToolError,
    PairKind,
    catalog,
    fetch_cedar_archive,
    parse_member_path,
    read_members,
    resolve_cache_dir,
    select_balanced_slice,
)
from typevet_evals.datasets.lfw import LfwChecksumError

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cedar"
MEMBERS = tuple(
    (FIXTURES / "members_excerpt.txt").read_text(encoding="utf-8").splitlines()
)
SLICE_IDS = tuple(
    (FIXTURES / "default_slice_ids.txt").read_text(encoding="utf-8").splitlines()
)


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


class FakeUnrar:
    """Stand-in for ``unrar x``: writes synthetic images for listed members."""

    def __init__(self, images: dict[str, bytes], *, failure: int = 0) -> None:
        """Store the synthetic archive content and a forced exit status."""
        self.images = images
        self.failure = failure
        self.commands: list[list[str]] = []

    def __call__(
        self, command: Sequence[str], *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        """Extract every listed member that the fake archive holds.

        Like ``unrar`` 7, the exit status is 10 when no listed member is in
        the archive. With ``check`` the fake raises as ``subprocess.run``.
        """
        assert capture_output is True
        self.commands.append(list(command))
        list_file = Path(command[-2].removeprefix("@"))
        dest = Path(command[-1])
        extracted = 0
        for name in list_file.read_text(encoding="utf-8").splitlines():
            if name in self.images and not self.failure:
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.images[name])
                extracted += 1
        status = self.failure or (0 if extracted else 10)
        if check and status:
            raise subprocess.CalledProcessError(status, list(command))
        return subprocess.CompletedProcess(list(command), status, b"", b"")


def _serving(payload: bytes, seen: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_source_constants_pin_the_cedar_archive() -> None:
    assert ARCHIVE_URL == "https://cedar.buffalo.edu/NIJ/data/signatures.rar"
    assert SOURCE_PAGE_URL == "https://cedar.buffalo.edu/NIJ/publications.html"
    assert ARCHIVE_SHA256 == (
        "f74b859352783b82399c1be48078b79ad637160ba11f16baf92911dd5568f4d6"
    )
    assert ARCHIVE_FILE_NAME == "signatures.rar"
    assert CACHE_ENV_VAR == "TYPEVET_CEDAR_CACHE"
    assert UNRAR_COMMAND == "unrar"
    assert (WRITER_COUNT, SAMPLES_PER_WRITER) == (55, 24)
    assert (DEFAULT_PER_KIND, DEFAULT_SEED) == (60, 0)


def test_member_paths_follow_the_archive_layout() -> None:
    genuine = CedarSignature(writer=10, sample=1, forged=False)
    forged = CedarSignature(writer=55, sample=24, forged=True)
    assert genuine.member_path == "signatures/full_org/original_10_1.png"
    assert forged.member_path == "signatures/full_forg/forgeries_55_24.png"
    assert genuine.signature_id == "original_10_1"
    assert forged.signature_id == "forgeries_55_24"


def test_parse_member_path_reads_the_fixture_listing() -> None:
    parsed = [parse_member_path(name) for name in MEMBERS]
    signatures = [item for item in parsed if item is not None]
    assert signatures == [
        CedarSignature(10, 1, forged=True),
        CedarSignature(10, 10, forged=True),
        CedarSignature(55, 24, forged=True),
        CedarSignature(1, 1, forged=False),
        CedarSignature(10, 1, forged=False),
        CedarSignature(55, 24, forged=False),
    ]
    for signature, name in zip(
        signatures, [n for n in MEMBERS if n.endswith(".png")], strict=True
    ):
        assert signature.member_path == name


@pytest.mark.parametrize(
    "name",
    [
        "signatures/full_org/forgeries_1_1.png",
        "signatures/full_forg/original_1_1.png",
        "signatures/full_org/original_0_1.png",
        "signatures/full_org/original_1_0.png",
    ],
)
def test_parse_member_path_rejects_a_layout_mismatch(name: str) -> None:
    with pytest.raises(ValueError, match="CEDAR"):
        parse_member_path(name)


def test_catalog_lists_every_signature_once() -> None:
    signatures = catalog()
    assert len(signatures) == 2 * WRITER_COUNT * SAMPLES_PER_WRITER
    assert len(set(signatures)) == len(signatures)
    counts = Counter((item.writer, item.forged) for item in signatures)
    assert set(counts.values()) == {SAMPLES_PER_WRITER}
    assert {writer for writer, _ in counts} == set(range(1, WRITER_COUNT + 1))


def test_default_slice_is_balanced_and_spread_over_writers() -> None:
    pairs = select_balanced_slice()
    assert len(pairs) == 3 * DEFAULT_PER_KIND
    assert len(set(pairs)) == len(pairs)
    assert Counter(pair.kind for pair in pairs) == dict.fromkeys(
        PairKind, DEFAULT_PER_KIND
    )
    for kind in PairKind:
        writers = Counter(pair.reference.writer for pair in pairs if pair.kind is kind)
        assert len(writers) == WRITER_COUNT
        assert max(writers.values()) == 2
    for pair in pairs:
        assert pair.reference.forged is False
        assert pair.reference != pair.questioned
        if pair.kind is PairKind.GENUINE_GENUINE:
            assert pair.same_writer is True
            assert pair.questioned.writer == pair.reference.writer
            assert pair.questioned.forged is False
        elif pair.kind is PairKind.GENUINE_SKILLED:
            assert pair.same_writer is False
            assert pair.questioned.writer == pair.reference.writer
            assert pair.questioned.forged is True
        else:
            assert pair.same_writer is False
            assert pair.questioned.writer != pair.reference.writer
            assert pair.questioned.forged is False
    assert len({pair.kind for pair in pairs[:20]}) == 3


def test_default_slice_matches_the_pinned_ids() -> None:
    assert tuple(pair.pair_id for pair in select_balanced_slice()) == SLICE_IDS


def test_slice_is_deterministic_and_seeded() -> None:
    signatures = catalog(writers=12, samples=6)
    first = select_balanced_slice(signatures, per_kind=10)
    again = select_balanced_slice(tuple(reversed(signatures)), per_kind=10)
    other = select_balanced_slice(signatures, per_kind=10, seed=1)
    assert first == again
    assert first != other


def test_slice_goes_deeper_when_writers_are_few() -> None:
    signatures = catalog(writers=3, samples=6)
    pairs = select_balanced_slice(signatures, per_kind=6)
    assert len(set(pairs)) == 18
    for kind in PairKind:
        writers = Counter(pair.reference.writer for pair in pairs if pair.kind is kind)
        assert writers == dict.fromkeys((1, 2, 3), 2)


def test_pair_id_and_member_paths() -> None:
    pair = CedarPair(
        PairKind.GENUINE_SKILLED,
        CedarSignature(7, 1, forged=False),
        CedarSignature(7, 4, forged=True),
    )
    assert pair.pair_id == "genuine_skilled:original_7_1:forgeries_7_4"
    assert pair.member_paths == (
        "signatures/full_org/original_7_1.png",
        "signatures/full_forg/forgeries_7_4.png",
    )


def _only(writers: int, genuine: int, forged: int) -> tuple[CedarSignature, ...]:
    return tuple(
        CedarSignature(writer, sample, forged=is_forged)
        for writer in range(1, writers + 1)
        for is_forged, count in ((False, genuine), (True, forged))
        for sample in range(1, count + 1)
    )


def test_slice_skips_a_writer_whose_skilled_or_random_images_run_out() -> None:
    rich = [CedarSignature(1, n, forged=False) for n in range(1, 7)]
    rich += [CedarSignature(1, n, forged=True) for n in (1, 2)]
    poor = [CedarSignature(2, 1, forged=False)]
    poor += [CedarSignature(2, n, forged=True) for n in (1, 2)]
    pairs = select_balanced_slice([*rich, *poor], per_kind=2)
    skilled = [pair for pair in pairs if pair.kind is PairKind.GENUINE_SKILLED]
    assert {pair.reference.writer for pair in skilled} == {1, 2}
    with pytest.raises(ValueError, match="only 2 genuine_random pairs"):
        select_balanced_slice([*rich, *poor], per_kind=3)
    short = [*rich[:6], CedarSignature(1, 1, forged=True), *poor]
    with pytest.raises(ValueError, match="only 2 genuine_skilled pairs"):
        select_balanced_slice(short, per_kind=3)


@pytest.mark.parametrize(
    ("signatures", "per_kind", "match"),
    [
        (_only(5, 3, 1), 0, "per_kind"),
        (_only(2, 3, 1), 4, "genuine_genuine"),
        (_only(4, 1, 1), 2, "genuine_genuine"),
        (_only(4, 3, 0), 2, "genuine_skilled"),
        (_only(1, 9, 9), 1, "genuine_random"),
    ],
)
def test_slice_refuses_a_shortfall(
    signatures: tuple[CedarSignature, ...], per_kind: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        select_balanced_slice(signatures, per_kind=per_kind)


def test_resolve_cache_dir_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "env"))
    assert resolve_cache_dir(tmp_path / "arg") == tmp_path / "arg"
    assert resolve_cache_dir() == tmp_path / "env"
    monkeypatch.delenv(CACHE_ENV_VAR)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_cache_dir() == tmp_path / ".cache" / "typevet" / "cedar"


def test_fetch_downloads_once_and_refuses_another_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic archive bytes"
    monkeypatch.setattr(cedar, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest())
    seen: list[str] = []
    with _serving(payload, seen) as client:
        path = fetch_cedar_archive(cache_dir=tmp_path, client=client)
        again = fetch_cedar_archive(cache_dir=tmp_path, client=client)
    assert path == again == tmp_path / ARCHIVE_FILE_NAME
    assert path.read_bytes() == payload
    assert seen == [ARCHIVE_URL]

    other = tmp_path / "other"
    with _serving(b"tampered", []) as client, pytest.raises(LfwChecksumError):
        fetch_cedar_archive(cache_dir=other, client=client)
    assert not (other / ARCHIVE_FILE_NAME).exists()


def test_fetch_refuses_a_cached_file_with_another_hash(tmp_path: Path) -> None:
    (tmp_path / ARCHIVE_FILE_NAME).write_bytes(b"not the archive")
    seen: list[str] = []
    with _serving(b"unused", seen) as client, pytest.raises(LfwChecksumError):
        fetch_cedar_archive(cache_dir=tmp_path, client=client)
    assert seen == []


def test_read_members_extracts_requested_members(tmp_path: Path) -> None:
    left = CedarSignature(3, 2, forged=False).member_path
    right = CedarSignature(3, 5, forged=True).member_path
    images = {left: _solid_png((255, 0, 0)), right: _solid_png((0, 0, 255))}
    fake = FakeUnrar(images)
    archive = _archive(tmp_path)
    found = read_members(
        archive, [left, right, left], unrar="/opt/fake/unrar", run=fake
    )
    assert found == images
    (command,) = fake.commands
    assert command[:4] == ["/opt/fake/unrar", "x", "-inul", "-y"]
    assert command[4] == str(archive)


def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / ARCHIVE_FILE_NAME
    archive.write_bytes(b"synthetic archive placeholder")
    return archive


def test_read_members_reports_a_missing_member(tmp_path: Path) -> None:
    fake = FakeUnrar({})
    missing = CedarSignature(9, 9, forged=False).member_path
    with pytest.raises(KeyError, match="original_9_9"):
        read_members(_archive(tmp_path), [missing], unrar="unrar", run=fake)
    assert len(fake.commands) == 1


def test_read_members_reports_missing_members_next_to_found_ones(
    tmp_path: Path,
) -> None:
    present = CedarSignature(1, 1, forged=False).member_path
    missing = CedarSignature(99, 1, forged=False).member_path
    fake = FakeUnrar({present: _solid_png((0, 255, 0))})
    with pytest.raises(KeyError, match="original_99_1"):
        read_members(_archive(tmp_path), [present, missing], unrar="u", run=fake)


def test_read_members_raises_other_tool_failures(tmp_path: Path) -> None:
    fake = FakeUnrar({}, failure=3)
    member = CedarSignature(1, 1, forged=False).member_path
    with pytest.raises(subprocess.CalledProcessError) as caught:
        read_members(_archive(tmp_path), [member], unrar="u", run=fake)
    assert caught.value.returncode == 3


def test_read_members_refuses_a_missing_archive(tmp_path: Path) -> None:
    fake = FakeUnrar({})
    member = CedarSignature(1, 1, forged=False).member_path
    with pytest.raises(FileNotFoundError, match=r"absent\.rar"):
        read_members(tmp_path / "absent.rar", [member], unrar="u", run=fake)
    assert fake.commands == []


@pytest.mark.parametrize(
    "name",
    [
        "../../../../../../etc/hostname",
        "signatures/full_org/../../../etc/hostname",
        "/etc/hostname",
        "signatures/Readme.txt",
        "signatures/full_org/original_1_1.png/../../x",
    ],
)
def test_read_members_refuses_a_non_signature_path(tmp_path: Path, name: str) -> None:
    fake = FakeUnrar({})
    ok = CedarSignature(1, 1, forged=False).member_path
    with pytest.raises(ValueError, match="not a CEDAR signature member"):
        read_members(_archive(tmp_path), [ok, name], unrar="u", run=fake)
    assert fake.commands == []


def test_read_members_refuses_without_an_unrar_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cedar.shutil, "which", lambda _name: None)
    with pytest.raises(CedarToolError, match="unrar"):
        read_members(tmp_path / "a.rar", ["signatures/full_org/original_1_1.png"])
