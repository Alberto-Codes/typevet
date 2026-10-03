"""Module command validates its public run options."""

import pytest

from typevet_evals.jevbench_run import main

pytestmark = pytest.mark.unit


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Help exposes every required input and evidence destination."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for flag in (
        "--tasks",
        "--model",
        "--results",
        "--raw-dir",
        "--ledger",
        "--manifest",
        "--limit",
        "--cap-usd",
        "--reserve-usd",
    ):
        assert flag in output


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
@pytest.mark.parametrize("flag", ["--cap-usd", "--reserve-usd"])
def test_invalid_budget_rejected_before_manifest(
    tmp_path, value: str, flag: str
) -> None:
    """Nonfinite or negative controls cannot produce malformed evidence."""
    args = []
    for name in ("tasks", "results", "raw-dir", "ledger", "manifest"):
        args.extend([f"--{name}", str(tmp_path / name)])
    with pytest.raises(SystemExit) as exit_info:
        main([*args, "--model", "fake", flag, value])
    assert exit_info.value.code == 2
    assert not (tmp_path / "manifest").exists()
