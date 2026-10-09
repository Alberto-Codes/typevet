# Install typevet

Kind: how-to.

Use this page to install typevet into your own project.

## Prerequisites

- Python 3.12 or newer. The package declares `requires-python = ">=3.12"`.
- [uv](https://docs.astral.sh/uv/) or pip.

An older Python rejects the package. uv reports that typevet depends on `Python>=3.12`.

## Install from PyPI

In your project directory, run one of these commands:

```bash
uv add typevet
```

```bash
pip install typevet
```

The installer also installs the runtime dependencies: `httpx`, `jsonschema` and `structlog`.

## Install from a checkout

Use this procedure to test a commit that is not on PyPI.
You need a clone of the typevet repository.

### Build the wheel

From the typevet checkout, run:

```bash
cd /path/to/typevet
uv build --wheel --out-dir /tmp/typevet-dist
```

Expected output ends with this line:

```text
Successfully built /tmp/typevet-dist/typevet-0.2.0-py3-none-any.whl
```

The version in the file name comes from `[project].version` in `pyproject.toml`.
Replace `0.2.0` in the next steps if your checkout has a different version.

### Install the wheel into a fresh virtual environment

In your own project directory, run:

```bash
uv venv
uv pip install /tmp/typevet-dist/typevet-0.2.0-py3-none-any.whl
```

uv also installs the runtime dependencies: `httpx`, `jsonschema` and `structlog`.

Alternatively, add the wheel to an existing uv project:

```bash
uv add /tmp/typevet-dist/typevet-0.2.0-py3-none-any.whl
```

## Install an optional extra

The package declares two extras.

The `cli` extra installs Typer.
typevet does not yet ship a console script, so the extra adds no command.

```bash
uv add "typevet[cli]"
```

The `judgevet` extra installs `judgevet>=0.17,<0.19`.
It lets judgevet use typevet as its judgment provider.
See [use typevet as a judgevet provider](use-typevet-as-a-judgevet-provider.md).

```bash
uv add "typevet[judgevet]"
```

A bare install never imports judgevet.

## Check the installation

From your project directory, run:

```bash
uv run python -c "import typevet; print(typevet.__version__)"
```

Expected output:

```text
0.2.0
```

`typevet.__version__` reads the version from the installed distribution metadata.

## Next steps

- Make a first call with no network in the [offline tutorial](../tutorials/first-typed-judgment-offline.md).
- Pick an adapter in [Call typevet from Python](call-typevet-from-python.md).
- See the public import paths in [supported imports](../reference/supported-imports.md).
