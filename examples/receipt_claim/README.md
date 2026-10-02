# Receipt claim

Kind: reference. This page gives the command and the output shape of the receipt claim example.

The script reads one expense claim and its receipt image from `tests/fixtures/cord/expense_smoke/`. One Choice asks whether the image supports the claimed total.

Run from the repository root:

```console
uv run python examples/receipt_claim/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. It needs a router that serves a Gemma 4 model, named by `TYPEVET_LLAMA__MULTIMODAL_MODEL`. `TYPEVET_BACKEND=fake` runs offline with uniform answers. The script writes no file.

The output has the claim, each option with its probability, and the winner on the last line:

```text
claim: Expense claim for this receipt: total 80,500.
supported: <probability>
contradicted: <probability>
insufficient_evidence: <probability>
winner: <label>
```
