# Screenshot UI

Kind: reference. This page gives the command and the output shape of the screenshot UI example.

The script reads one browser screenshot from `tests/fixtures/psai/vision_smoke/`. The loader computes the SHA-256 digest of the PNG file and compares it with the `sha256` value in `manifest.json`. A different digest stops the script with a `ValueError`. One Choice asks which website chrome (logo, header and navigation) the screenshot shows. The options are `fox_news`, `home_depot`, `squarespace` and `other`. The manifest marks this screenshot as `fox_news`.

The screenshots come from the PSAI dataset under the MIT License. See the [PSAI fixture licence](../../tests/fixtures/psai/LICENSE.md).

Run from the repository root:

```console
uv run python examples/screenshot_ui/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. It needs a router that serves a Gemma 4 model, named by `TYPEVET_LLAMA__MULTIMODAL_MODEL`. `TYPEVET_BACKEND=fake` runs offline with uniform answers. The script writes no file.

The output has the state text, each option with its probability, and the winner on the last line:

```text
state: This is a screenshot of a web browser.
fox_news: <probability>
home_depot: <probability>
squarespace: <probability>
other: <probability>
winner: <label>
```
