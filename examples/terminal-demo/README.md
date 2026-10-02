# Terminal demo

Kind: reference. This page gives the command and the output shape of the terminal demo.

The script judges one customer message with a Noul, a Choice and a Score. It then checks one expense claim against two receipt images from `tests/fixtures/cord/expense_smoke/`. Last, it shows that bad inputs are refused before any HTTP call.

Run from the repository root:

```console
uv run python examples/terminal-demo/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp` and needs a router that serves a Gemma 4 model, named by `TYPEVET_LLAMA__MULTIMODAL_MODEL`. `TYPEVET_BACKEND=fake` runs offline with uniform answers. The script prints each probability distribution and writes a JSON receipt under `typevet-receipts/`.
