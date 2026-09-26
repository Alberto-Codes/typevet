# Install the private consumer bridge

Kind: how-to.

This distribution is private. Do not publish it.
The bridge supports synchronous text Noul, Choice and Score questions.
Choice accepts 2 to 24 ordered labels with string descriptions or `None`.
Score accepts 2 to 24 ordered string levels. Its legend starts at zero.
Raw mappings and consumer question objects use the same validation.
Nested criteria and instructions are unsupported.
The bridge preserves distributions, confidence and the probability-weighted Score value.

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


After installation, use the consumer question and policy types:

```python
from judgevet import Noul
from typevet_consumer_bridge import BridgeSettings, open_typevet_system_one

settings = BridgeSettings(
    base_url="http://localhost:8080",
    timeout=30.0,
    multimodal_model="your-served-model",
)
with open_typevet_system_one(settings=settings) as adapter:
    response = adapter.system_one(
        "The invoice total is 42 dollars.",
        {"total_is_42": Noul(instructions="Does the total equal 42 dollars?")},
        "your-served-model",
    )
```

This example makes a service call. The existing runtime requires a vision-capable
native Gemma service even for text. The bridge passes text unchanged, preserves
IDs and instructions, and returns new consumer answer objects. It validates the
whole request before judgment IO. Factory entry still performs metadata probes.
The response model is the requested routing identity, not an attested weight identity.
Unknown usage remains `None`. It cannot establish token or currency spend.

The context closes runtime-owned resources on every exit. A supplied HTTP client
remains caller-owned. `TypevetSystemOneAdapter(session)` borrows an open public
runtime session; its `close()` ends adapter access only. Closed adapters reject calls.
Known errors use bridge error subclasses with fixed messages. Unexpected errors
propagate. This does not provide a redactor, audit sink, retry policy or spend cap.
Callers own input redaction, external audit records and resource budgets.

Offline tests exercise both passing and failing consumer policy outcomes through
the real public factory with `httpx.MockTransport`. They do not establish live
service readiness, confidence calibration or general model quality.
