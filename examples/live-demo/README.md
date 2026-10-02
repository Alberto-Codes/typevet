# Live demo

Kind: reference. This page lists the files, the prerequisites, the command and the run of show for the live demo.

The page asks typed questions to Gemma 4 on a local llama.cpp router. It shows the probability of each answer. A "behind the scenes" panel shows the prompt, the candidate tokens and the router calls. The page uses the colour tokens of the documentation theme and follows the system colour scheme, light or dark.

## Files

| Path | Content |
|---|---|
| `server.py` | Web server. It holds one warm judgment session and serves the page and the API. |
| `bts.py` | Behind-the-scenes capture. It records the router calls and the scoring requests. |
| `index.html` | The web page. |
| `record.mjs` | Scripted headless Chrome recording of the run of show. |
| `insufficient-R01-blurred.png` | Blurred receipt. Upload it for the "insufficient evidence" case. |

## Prerequisites

- A llama.cpp router on `127.0.0.1:8090`. [Run Gemma 4 on llama.cpp](../../docs/how-to/run-gemma4-llamacpp.md) gives the steps.
- The router serves a Gemma 4 vision model. The default model id is `gemma-4-31b-kv9-q4km-mm`.
- The repository checkout, synced with `uv sync`. The gallery reads the six CORD receipts from `tests/fixtures/cord/expense_smoke/`.
- Pillow, for the gallery thumbnails. The `typevet-evals` dev group installs it.
- An NVIDIA GPU is optional. Without `/usr/bin/nvidia-smi`, the page shows the GPU as `unknown`.

## Command

Run the command from the repository root:

```bash
uv run python examples/live-demo/server.py
```

Wait for the `ready` line. The first start loads the model. This takes approximately 30 to 60 seconds. Then open `http://127.0.0.1:8765/` and press Ctrl+Shift+R.

The server reads these environment variables:

| Name | Default | Use |
|---|---|---|
| `LIVE_UI_PORT` | `8765` | Port of the web page. The server binds `127.0.0.1` only. |
| `TYPEVET_LLAMA__BASE_URL` | `http://127.0.0.1:8090` | llama.cpp router URL. |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | `gemma-4-31b-kv9-q4km-mm` | Router model id. The model must accept images. |
| `TYPEVET_LLAMA__TIMEOUT` | `900` | Router timeout in seconds. |

Each judgment writes one JSON receipt under `typevet-receipts/` in the repository root. Git ignores that directory.

## Run of show

1. Click "True total for R01", then "Ask Gemma 4". The answer is "supported".
2. Click "Same claim, swapped receipt", then "Ask Gemma 4". The answer is "contradicted".
3. Upload `insufficient-R01-blurred.png`, then click "Ask Gemma 4". The answer is "insufficient evidence".
4. Open the "behind the scenes" panel under the answer.
5. Select the Text tab, then click "Ask Gemma 4". Three typed answers show: yes or no, pick one, and a score.
6. Select the Image tab and click "Typed guarantee: try a GIF". typevet refuses the GIF before any model call.

Do not show a receipt with the total cut off. On that image, Gemma 4 answered "contradicted", not "insufficient". Issue [#189](https://github.com/Alberto-Codes/typevet/issues/189) tracks this concern. On the six-option fraud question, the panel shows a large off-menu mass. Issue [#207](https://github.com/Alberto-Codes/typevet/issues/207) tracks it.

## Recording

`record.mjs` drives the page through the Chrome DevTools Protocol and captures screencast frames. Start the server first. Then run the recorder from the repository root:

```bash
node examples/live-demo/record.mjs
```

Add `--dry` to check the overlay and the capture without model calls. The recorder reads these environment variables:

| Name | Default | Use |
|---|---|---|
| `TYPEVET_DEMO_UPLOAD` | `insufficient-R01-blurred.png` beside `record.mjs` | Image for the upload step. |
| `TYPEVET_DEMO_CHROME` | `google-chrome` on `PATH` | Chrome binary. |

The recorder writes its frames, its frame list and its log beside `record.mjs`. Join the frames with `ffmpeg` and the `frames.ffconcat` list.

## Related

- [`examples/terminal-demo/run.py`](../terminal-demo/run.py) is the terminal version of the same demo.
- [Examples](../README.md) lists every example.
