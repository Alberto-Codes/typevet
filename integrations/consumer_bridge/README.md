# Install the private consumer bridge

Kind: how-to.

This distribution is private. Do not publish it.
The current scaffold proves packaging only. It does not expose an adapter API yet.

1. Obtain both wheels identified by `dependency-artifacts.json`.
2. Run this command from the Typevet checkout with the wheel paths:

```bash
uv run python integrations/consumer_bridge/scripts/verify_install.py \
  --typevet-wheel /operator/artifacts/typevet-0.1.0-py3-none-any.whl \
  --consumer-wheel /operator/artifacts/judgevet-0.13.0-py3-none-any.whl
```

The verifier checks both hashes before installation. It builds the bridge wheel in a temporary directory.
It installs each dependency wheel explicitly into an isolated environment outside the checkout.
It guards the installed bridge import against client creation, network calls, and package metadata queries.
It checks import paths against installed RECORD entries and prints installed versions.
It also installs base Typevet alone and checks that neither companion nor consumer is present.
Temporary environments are removed on exit.

Do not substitute registry artifacts with matching version numbers.
Do not use editable sibling checkouts as dependency evidence.
The manifest identifies the accepted private dependency artifacts.
The printed bridge hash identifies the wheel built from the current source.

For a retained wheel, build into an external directory:

```bash
uv build integrations/consumer_bridge --wheel --out-dir /operator/artifacts/bridge
```

Keep the wheel and its SHA256 with the acceptance receipt.
Packaging checks do not prove consumer policy behavior or live model quality.
