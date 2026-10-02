"""Contract: every ``examples/*/run.py`` runs offline on the fake backend (#395).

Each example loads by file path into a fresh module and runs ``main()`` in
process with ``TYPEVET_BACKEND=fake``. A ``TYPEVET_FAKE__DISTRIBUTIONS`` file
scripts the ``verdict`` winner. Each example runs with two winners, so neither a
uniform default nor one fixed label can pass. Each example has one row in
``EXPECTED`` that names the output line that proves the winner.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
# Two scripted winners. Both differ from the uniform default (``supported``), so
# an example that prints one fixed label fails one case.
WINNERS = ("contradicted", "insufficient_evidence")

# Example directory -> (where to look, expected stdout text with ``{winner}``).
# "last": the last stdout line equals the text. "part": some line holds it.
EXPECTED: dict[str, tuple[str, str]] = {
    "receipt_claim": ("last", "winner: {winner}"),
    "terminal-demo": ("part", "-> model answer: {winner} ("),
}
NAMES = sorted({p.parent.name for p in EXAMPLES.glob("*/run.py")} | set(EXPECTED))


@pytest.mark.parametrize("winner", WINNERS)
@pytest.mark.parametrize("name", NAMES)
def test_example_prints_the_scripted_winner(
    name: str,
    winner: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert name in EXPECTED, f"add a row for examples/{name}/ to EXPECTED"
    script = EXAMPLES / name / "run.py"
    assert script.is_file(), f"missing {script.relative_to(REPO)}"
    dist = tmp_path / "distributions.json"
    dist.write_text(json.dumps({"verdict": {winner: 1.0}}))
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("TYPEVET_BACKEND", "fake")
    monkeypatch.setenv("TYPEVET_FAKE__DISTRIBUTIONS", str(dist))

    spec = importlib.util.spec_from_file_location(f"example_{name}", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "OUT_DIR"):
        monkeypatch.setattr(module, "OUT_DIR", tmp_path / "typevet-receipts")
    module.main()

    lines = capsys.readouterr().out.splitlines()
    where, template = EXPECTED[name]
    text = template.format(winner=winner)
    if where == "last":
        assert lines[-1] == text
    else:
        assert any(text in line for line in lines)
