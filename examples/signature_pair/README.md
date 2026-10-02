# Signature pair

Kind: reference. This page gives the command and the output shape of the signature pair example.

The script reads two image files from the paths in `TYPEVET_EXAMPLE_IMAGE_A` and `TYPEVET_EXAMPLE_IMAGE_B`. The files must be `.png`, `.jpg` or `.jpeg`. Image A is the reference signature and image B is the questioned signature. One Noul asks whether the same person signed both. The state text and the Noul come from the `typevet_evals` workspace member.

The model answers about appearance. It makes no identity or authenticity claim. See [verified evidence and inferred claims](../../docs/explanation/verification.md).

Run from the repository root:

```console
TYPEVET_EXAMPLE_IMAGE_A=a.png TYPEVET_EXAMPLE_IMAGE_B=b.png uv run python examples/signature_pair/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. It needs a router that serves a Gemma 4 model, named by `TYPEVET_LLAMA__MULTIMODAL_MODEL`. `TYPEVET_BACKEND=fake` runs offline with uniform answers. A missing variable or file stops the script with a `ValueError` that names the variable. The script writes no file.

The output has the state text, the probability of yes, and the winner on the last line. The winner is `yes` when the probability is more than 0.5, and `no` in other cases:

```text
state: Image 1 and image 2 each show one handwritten signature. ...
P(yes): <probability>
winner: <yes or no>
```
