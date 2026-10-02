# Examples

Kind: reference. This page lists each runnable example under `examples/` and the one command that runs it.

Each scenario example holds one `README.md` and one `run.py`. The live demo is a web page with its own `README.md`. Run every command from the repository root. Epic [#385](https://github.com/Alberto-Codes/typevet/issues/385) tracks the tree.

| Scenario | Judgment kind | One command | Needs a server |
|---|---|---|---|
| `calibrate/` | Noul (fraud message), calibrated with a Platt map | `uv run python examples/calibrate/run.py` | fake only (`TYPEVET_BACKEND=fake`); the fixture map is synthetic and names the fake model |
| `check_register/` | Choice (cheque image against a register row) | `uv run python examples/check_register/run.py` | yes (llama.cpp with Gemma 4), or no with `TYPEVET_BACKEND=fake` |
| `live-demo/` | Choice (receipt claim against a receipt image) | `uv run python examples/live-demo/server.py` | yes (llama.cpp with Gemma 4) |
| `receipt_claim/` | Choice (receipt claim against a receipt image) | `uv run python examples/receipt_claim/run.py` | yes (llama.cpp with Gemma 4), or no with `TYPEVET_BACKEND=fake` |
| `terminal-demo/` | Noul, Choice and Score (customer message), Choice (receipt image) | `uv run python examples/terminal-demo/run.py` | yes (llama.cpp with Gemma 4), or no with `TYPEVET_BACKEND=fake` |
