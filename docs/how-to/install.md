# Install typevet

Kind: how-to.

Use this page to install typevet into a project outside the typevet checkout.
Today you build a wheel from the checkout and install that wheel.

## Prerequisites

- Python 3.12 or newer. The package declares `requires-python = ">=3.12"`.
- [uv](https://docs.astral.sh/uv/) on your `PATH`.
- A clone of the typevet repository.

An older Python rejects the wheel. uv reports that typevet depends on `Python>=3.12`.

## Build the wheel

From the typevet checkout, run:

```bash
cd /path/to/typevet
uv build --wheel --out-dir /tmp/typevet-dist
```

Expected output ends with this line:

```text
Successfully built /tmp/typevet-dist/typevet-0.1.0-py3-none-any.whl
```

The version in the file name comes from `[project].version` in `pyproject.toml`.
Replace `0.1.0` in the next steps if your checkout has a different version.

## Install the wheel into a fresh virtual environment

In your own project directory, run:

```bash
uv venv
uv pip install /tmp/typevet-dist/typevet-0.1.0-py3-none-any.whl
```

uv also installs the runtime dependencies: `httpx`, `jsonschema` and `structlog`.

Alternatively, add the wheel to an existing uv project:

```bash
uv add /tmp/typevet-dist/typevet-0.1.0-py3-none-any.whl
```

## Install the optional extra

The package declares one extra, `cli`. It installs Typer.
typevet does not yet ship a console script, so the extra adds no command.

```bash
uv pip install "/tmp/typevet-dist/typevet-0.1.0-py3-none-any.whl[cli]"
```

## Install from PyPI

typevet is not yet published on PyPI.
The release task [#255](https://github.com/Alberto-Codes/typevet/issues/255) owns the first publication.
Until then, `pip install typevet` and `uv add typevet` fail.

## Check the installation

From your project directory, run:

```bash
uv run python -c "import typevet; print(typevet.__version__)"
```

Expected output:

```text
0.1.0
```

`typevet.__version__` reads the version from the installed distribution metadata.

## Next steps

- Make a first call with no network in the [offline tutorial](../tutorials/first-typed-judgment-offline.md).
- Pick an adapter in [Call typevet from Python](call-typevet-from-python.md).
- See the public import paths in [supported imports](../reference/supported-imports.md).
