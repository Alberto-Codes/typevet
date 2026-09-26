# Verify package typing and version

Kind: how-to, for maintainers.

Run these checks before publishing a wheel. For supported consumer imports, see
[supported imports](../reference/supported-imports.md).

The `typevet/py.typed` marker declares inline typing support under
[PEP 561](https://peps.python.org/pep-0561/). Its presence does not prove that
every use of the library is type-correct. Check both the artifact and a typed
consumer.

`typevet.__version__` reads `[project].version` from the installed distribution
via `importlib.metadata`. Authority stays in `pyproject.toml`.

## Build and inspect the artifact

Run this Bash procedure from a checkout with Python and uv installed. It
builds into a new temporary directory so stale wheels cannot be selected.
The temporary directory remains available for inspection after the check.

```bash
artifact_dir=$(mktemp -d)
uv build --out-dir "$artifact_dir"
set -- "$artifact_dir"/typevet-*.whl
test "$#" -eq 1 && test -f "$1" || exit 1
wheel_path=$1
uv run python - "$wheel_path" <<'PYCODE'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    assert "typevet/py.typed" in wheel.namelist()
print("wheel contains typevet/py.typed")
PYCODE
```

Expected output includes `wheel contains typevet/py.typed`.

## Check an isolated consumer

Continue in the same shell. The environment is outside the checkout. The
consumer imports the installed package and checks a supported root type.

```bash
uv venv "$artifact_dir/consumer"
consumer_python="$artifact_dir/consumer/bin/python"
uv pip install --python "$consumer_python" "$wheel_path"
cat > "$artifact_dir/consumer_check.py" <<'PYCODE'
from typevet import GenerationRequest

request = GenerationRequest(prompt="hi", schema={"type": "object"}, model="fake")
assert request.model == "fake"
PYCODE
(cd "$artifact_dir" && "$consumer_python" consumer_check.py)
uv run ty check --project "$artifact_dir" --python "$consumer_python" "$artifact_dir/consumer_check.py"
(cd "$artifact_dir" && "$consumer_python" - <<'PYCODE'
import sysconfig
from importlib.metadata import version
from pathlib import Path

import typevet

assert typevet.__version__ == version("typevet")
location = Path(typevet.__file__).resolve()
assert location.is_relative_to(Path(sysconfig.get_paths()["purelib"]).resolve())
assert location.with_name("py.typed").is_file()
print("installed package, version, and typing marker verified")
PYCODE
)
```

Expected output includes `All checks passed!` and
`installed package, version, and typing marker verified`. `ty check` takes file
paths; its `-c` option sets configuration and does not execute Python source.
A missing export, invalid attribute or incompatible assignment must be fixed
before publication. Marker inspection and consumer checking prove different
properties; a type check alone does not prove the marker is packaged.

## Fast unit check in CI

```bash
uv run pytest tests/unit/test_package_wheel.py -q
```
