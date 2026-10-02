# Check register

Kind: reference. This page gives the command and the output shape of the check register example.

The script renders one synthetic SPECIMEN cheque and its register row from the `typevet_evals` workspace member. The payee is an invented name. One Choice asks whether the cheque shows the payee and the amount of the register row.

Run from the repository root:

```console
uv run python examples/check_register/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. It needs a router that serves a Gemma 4 model, named by `TYPEVET_LLAMA__MULTIMODAL_MODEL`. `TYPEVET_BACKEND=fake` runs offline with uniform answers. The script writes no file.

The output has the register row, each option with its probability, and the winner on the last line:

```text
Register row: payee Lanternfield Electric; amount $1,188.15
matches: <probability>
differs: <probability>
unreadable: <probability>
winner: <label>
```
