"""Unit tests for the offline check contact-sheet command (#344)."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet_evals.check_match import SEED_ENV, check_match_seed
from typevet_evals.cli import check_sheets

pytestmark = pytest.mark.unit

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sheets(out: Path) -> list[Path]:
    return sorted(out.glob("*.png"))


def test_command_writes_one_grid_per_register_row_for_a_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "seed1"
    code = check_sheets.main(["--seed", "1", "--rows", "2", "--out", str(out)])
    assert code == 0
    sheets = _sheets(out)
    assert [p.name for p in sheets] == ["seed1_row00.png", "seed1_row01.png"]
    for sheet in sheets:
        data = sheet.read_bytes()
        assert data.startswith(PNG_SIGNATURE)
        assert len(data) > 1000
    printed = capsys.readouterr().out
    assert str(sheets[0]) in printed


def test_another_seed_gives_different_sheets(tmp_path: Path) -> None:
    for seed in ("1", "2"):
        code = check_sheets.main(
            ["--seed", seed, "--rows", "1", "--out", str(tmp_path)]
        )
        assert code == 0
    first = (tmp_path / "seed1_row00.png").read_bytes()
    second = (tmp_path / "seed2_row00.png").read_bytes()
    assert first != second


def test_same_seed_gives_the_same_bytes(tmp_path: Path) -> None:
    for name in ("a", "b"):
        check_sheets.main(["--rows", "1", "--out", str(tmp_path / name)])
    first = (tmp_path / "a" / "seed0_row00.png").read_bytes()
    assert first == (tmp_path / "b" / "seed0_row00.png").read_bytes()


@pytest.mark.parametrize("rows", ["0", "21"])
def test_row_count_outside_the_slice_is_a_usage_error(
    tmp_path: Path, rows: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = check_sheets.main(["--rows", rows, "--out", str(tmp_path)])
    assert code == 2
    assert "rows" in capsys.readouterr().err
    assert not _sheets(tmp_path)


def test_row_range_error_prints_the_usage_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A range error is a usage error, so it shows the usage line (#349).
    code = check_sheets.main(["--rows", "0", "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage:")
    assert "--rows must be 1 to 20: 0" in err


# Each string is either a seed that TYPEVET_CHECK_MATCH_SEED accepts or one it
# refuses; the flag must agree on each (#349).
_SEED_STRINGS = [
    "7",
    " 7 ",
    "007",
    "",
    "  ",
    "+1",
    "-1",
    "1_0",
    "1.0",
    "one",
    chr(0xFF11),  # fullwidth digit one
    chr(0x0663),  # Arabic-Indic digit three
]


@pytest.mark.parametrize("raw", _SEED_STRINGS, ids=ascii)
def test_seed_flag_accepts_exactly_the_strings_the_variable_accepts(
    tmp_path: Path, raw: str
) -> None:
    try:
        expected: int | None = check_match_seed({SEED_ENV: raw})
    except ValueError:
        expected = None
    code = check_sheets.main(["--seed", raw, "--rows", "1", "--out", str(tmp_path)])
    if expected is None:
        assert code == 2
        assert not _sheets(tmp_path)
    else:
        assert code == 0
        assert [p.name for p in _sheets(tmp_path)] == [f"seed{expected}_row00.png"]


@pytest.mark.parametrize(
    "argv",
    [["--seed", "-1"], ["--seed", "one"], []],
    ids=["negative", "text", "no-out"],
)
def test_bad_seed_or_missing_out_is_a_usage_error(
    tmp_path: Path, argv: list[str]
) -> None:
    out = ["--out", str(tmp_path)] if argv else []
    assert check_sheets.main([*argv, *out]) == 2
    assert not _sheets(tmp_path)
