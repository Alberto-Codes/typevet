"""Contract: isolated wheel exercises public native vision factory ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_gemma_native_vision_wheel_consumer.py
    ```
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gemma_native_vision_wheel_proof import run_offline_wheel_smoke
from typevet_evals.wheel_isolated import run_isolated_wheel_python


@pytest.mark.contract
def test_isolated_wheel_factory_smoke_uses_tmp_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Installed wheel calls ``open_gemma_native_vision_judgment`` without checkout imports."""
    fixture_consumer = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "consumer"
    )
    before = {
        p.name for p in fixture_consumer.glob("instruction-variant-receipt-*attempt*")
    }
    code, wheel = run_offline_wheel_smoke(work_dir=tmp_path)
    assert code == 0
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    checkout = Path(__file__).resolve().parents[3]
    assert set(receipt["module_paths"]) == {"smoke", "factory"}
    public_consumer = run_isolated_wheel_python(
        wheel=wheel,
        cwd=tmp_path / "isolated_cwd",
        extra_wheels=sorted((tmp_path / "evals-dist").glob("typevet_evals-*.whl")),
        source="""
import json
from pathlib import Path
import typevet.runtime as runtime
import typevet_evals.gemma_native_vision_wheel_smoke as smoke
assert runtime.open_gemma_native_vision_judgment is smoke.open_gemma_native_vision_judgment
smoke.open_gemma_native_vision_judgment = runtime.open_gemma_native_vision_judgment
print(json.dumps({"runtime": str(Path(runtime.__file__).resolve())}))
raise SystemExit(smoke.run_wheel_smoke())
""",
    )
    assert public_consumer.returncode == 0, public_consumer.stderr
    public_lines = public_consumer.stdout.splitlines()
    receipt["module_paths"].update(json.loads(public_lines[0]))
    assert json.loads(public_lines[-1])["factory_smoke"] == "ok"
    for module_path in receipt["module_paths"].values():
        installed = Path(module_path)
        assert "site-packages" in installed.parts
        assert not installed.is_relative_to(checkout)
    assert receipt["completion_calls"] == 2
    after = {
        p.name for p in fixture_consumer.glob("instruction-variant-receipt-*attempt*")
    }
    assert before == after
