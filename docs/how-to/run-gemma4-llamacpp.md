# Run Gemma 4 on local llama.cpp for typevet

Kind: how-to.

Use the bazzite-dotfiles llama.cpp router. The router serves GGUFs under
`~/models` on `127.0.0.1:8090`. See
`../bazzite-dotfiles/podman/llama-cpp/README.md` for install and tuning.

## Prerequisites

1. The user service is up:

   ```bash
   systemctl --user status llama-cpp-tuned.service
   curl -s http://127.0.0.1:8090/v1/models | jq '.data[].id'
   ```

2. Gemma 4 MVP weights are visible to the router as
   `gemma-4-31b-24gib-kv11-decoder` (GGUF hardlinked into `~/models`, preset in
   `~/models/presets.ini`). If the id is missing, restart the service after
   adding the file and preset.

## Call typevet

```bash
cd /path/to/typevet
uv sync
TYPEVET_GEMMA_MODEL=gemma-4-31b-24gib-kv11-decoder \
  uv run pytest -m live -q
```

Or from Python:

```python
from typevet.adapters.outbound import LlamaCppGenerationAdapter
from typevet.domain.models import GenerationRequest

schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
with LlamaCppGenerationAdapter() as port:
    result = port.generate(
        GenerationRequest(
            prompt="Return whether 2+2 equals 4.",
            schema=schema,
            model="gemma-4-31b-24gib-kv11-decoder",
        )
    )
print(result.value)
```

## Notes

- The router keeps `--models-max 1`. Loading Gemma 4 unloads the resident model.
- First load of a 13 GiB GGUF can take minutes. Raise the adapter timeout.
- Direction for presets and GPU image tags lives in bazzite-dotfiles, not here.
