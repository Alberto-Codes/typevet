# Scam message

Kind: reference. This page gives the command and the output shape of the scam message example.

The script reads line 2 of `tests/fixtures/difraud/sms_test_subset.jsonl`. That line has label 1, a scam. One Noul asks whether the message is a scam, phishing or social-engineering attempt. The user message is the suspect text itself. The Noul instructions come from the `typevet_evals` workspace member, so the example uses the same words as the DIFrauD evaluation.

The messages come from the DIFrauD dataset under the MIT License. Its SMS sources are under CC BY 4.0. See the [DIFrauD fixture licence](../../tests/fixtures/difraud/LICENSE.md).

Run from the repository root:

```console
uv run python examples/scam_message/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. It needs a llama.cpp server. `TYPEVET_BACKEND=fake` runs offline with uniform answers. The script writes no file.

The output has the message, the probability of yes, and the winner on the last line. The winner is `yes` when the probability is more than 0.5, and `no` in other cases:

```text
message: As a registered optin subscriber ur draw 4 £100 gift voucher will be entered on receipt of a correct ans to 80062 Whats No1 in the BBC charts
P(yes): <probability>
winner: <yes or no>
```
