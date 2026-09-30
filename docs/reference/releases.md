# Releases

Kind: reference.

This page states where typevet releases go, how the project cuts a version,
what a version before 1.0.0 means for you, and how to read the installed
version.

## Where releases are published

| Channel | Location | Source |
|---|---|---|
| PyPI | [pypi.org/project/typevet](https://pypi.org/project/typevet/) | `.github/workflows/publish.yml` |
| GitHub releases | [github.com/Alberto-Codes/typevet/releases](https://github.com/Alberto-Codes/typevet/releases) | `release-please-config.json` |
| Changelog | [`CHANGELOG.md`](https://github.com/Alberto-Codes/typevet/blob/main/CHANGELOG.md) | `release-please-config.json` (`changelog-path`) |
| TestPyPI | Development versions only, for rehearsal | `.github/workflows/test-publish.yml` |

The project publishes one distribution, `typevet`, with one version.
Release tags use the form `v<version>`, for example `v0.1.0`.
The config sets `include-component-in-tag` to `false`, so the tag has no package name.

For each published release, `publish.yml` makes these artifacts:

- One wheel and one source distribution. The workflow uploads both to PyPI.
- A signed GitHub build-provenance attestation over the wheel and the source distribution.
- A GitHub CycloneDX software bill of materials (SBOM) attestation over the wheel.

## How a version is cut

[release-please](https://github.com/googleapis/release-please) cuts each version from the commit history.
Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
See [commit messages](commits.md) for the type vocabulary.

1. A push to `main` runs `.github/workflows/release-please.yml`.
2. release-please opens or updates a release pull request.
   The pull request sets the new version in `pyproject.toml` and `uv.lock`.
   It also adds a section to `CHANGELOG.md`.
3. A maintainer merges the release pull request.
   release-please then creates a draft GitHub release, because the config sets `draft` to `true`.
4. A maintainer edits and publishes the draft release.
   GitHub creates the version tag at that moment.
5. The published release runs `.github/workflows/publish.yml`.
   Nothing reaches PyPI before this step.

Before the upload, `publish.yml` stops the release when any check fails:

- The tagged commit must be on `main`.
- The tag version must equal `[project].version` in `pyproject.toml`.
- The build must produce exactly one wheel.
- The wheel must pass `scripts/smoke_release.py` for the tag version.

The upload uses PyPI trusted publishing over OpenID Connect (OIDC).
The workflow uses no PyPI API token.

### Changelog sections

| Commit type | Changelog section |
|---|---|
| `feat` | Features |
| `fix` | Fixes |
| `perf` | Performance |
| `refactor` | Refactoring |
| `docs` | Documentation |
| `test`, `chore`, `ci`, `build` | Hidden |

A hidden type does not appear in `CHANGELOG.md`.
Do not edit changelog entries by hand.

## What pre-1.0 means

The current version is `0.1.0`, from `pyproject.toml` and `.release-please-manifest.json`.
The package declares the classifier `Development Status :: 3 - Alpha`.

- The config sets `bump-minor-pre-major` to `true`.
  Before 1.0.0, a breaking change raises the minor version, not the major version.
  For example, a breaking change after `0.1.0` gives `0.2.0`.
- A change in the minor version can therefore break your code.
  An upper bound such as `typevet>=0.1,<0.2` excludes the next minor release.
- Read the `CHANGELOG.md` section for each release before you upgrade.
- [Supported imports](supported-imports.md) lists the public surface and the 0.1.0 compatibility assessment.
  That page requires a compatibility note for each breaking rename or removed export.

No file in this repository states a support window, a service level agreement (SLA) or a deprecation period.
The project promises none of them.

## Read the installed version

```bash
python -c "import typevet; print(typevet.__version__)"
```

`typevet.__version__` reads the installed distribution metadata.
See [supported imports](supported-imports.md) for the mechanics and the checkout fallback.
