# Serve typevet on vLLM

Kind: how-to.

Use this page to run typevet generation and judgment against a vLLM
OpenAI-compatible server. The steps follow the one tested pin from the
[#170 live receipt](https://github.com/Alberto-Codes/typevet/issues/170#issuecomment-5884707915).

## Supported versions

typevet supports one tested pin. Other versions are not tested.

| Item | Tested value |
|---|---|
| Server image | `vllm/vllm-openai:v0.30.0` (`/version` returns `0.30.0`) |
| Model | `google/gemma-4-31B-it` |
| Model revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Weights | BF16, no quantization |
| GPU | One H100 with 80 GB of memory |
| Served model name | `gemma-4-31b-it` |

The tested server flags:

```text
--max-model-len 8192
--gpu-memory-utilization 0.95
--max-num-seqs 4
--limit-mm-per-prompt {"image":2}
--logprobs-mode raw_logprobs
--served-model-name gemma-4-31b-it
```

The server requires an API key. The server reads it from `VLLM_API_KEY`.

## Start the server

1. Optional: get a Hugging Face token. The Gemma 4 weights are not gated.
   The tested run supplied a token.

2. Start the stock image with about 150 GB of container disk. This example
   uses `docker run`. The tested run used the image's default entrypoint
   with the same server arguments. The `--gpus`, `--ipc` and `-p` options
   are standard Docker options that the tested run did not use.

   ```bash
   export VLLM_API_KEY='<your-key>'
   export HF_TOKEN='<your-hugging-face-token>'
   docker run --gpus all --ipc=host -p 8000:8000 \
     -e VLLM_API_KEY -e HF_TOKEN \
     vllm/vllm-openai:v0.30.0 \
     --model google/gemma-4-31B-it \
     --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.95 \
     --max-num-seqs 4 \
     --limit-mm-per-prompt '{"image":2}' \
     --logprobs-mode raw_logprobs \
     --served-model-name gemma-4-31b-it
   ```

3. Wait for the model to load. In the tested run, `/v1/models` answered
   about 7 minutes after the server started.

## Check the server

1. Check the server version:

   ```bash
   curl -s -H "Authorization: Bearer $VLLM_API_KEY" \
     http://127.0.0.1:8000/version
   ```

   Expect `{"version":"0.30.0"}`.

2. Check the served model name:

   ```bash
   curl -s -H "Authorization: Bearer $VLLM_API_KEY" \
     http://127.0.0.1:8000/v1/models | jq '.data[].id'
   ```

   Expect `"gemma-4-31b-it"`.

## Configure typevet

Set the backend and the server settings in the environment:

```bash
export TYPEVET_BACKEND=vllm
export TYPEVET_VLLM__BASE_URL=http://127.0.0.1:8000
export TYPEVET_VLLM__MODEL=gemma-4-31b-it
export TYPEVET_VLLM__API_KEY="$VLLM_API_KEY"
export TYPEVET_VLLM__TIMEOUT=300
```

[Configuration](../reference/configuration.md) lists each variable, its
default and its validation rule. These rules apply most often:

- `TYPEVET_BACKEND` accepts `llama_cpp` or `vllm`. An empty value selects
  `llama_cpp`.
- `TYPEVET_VLLM__BASE_URL` and `TYPEVET_VLLM__MODEL` are required.
- `TYPEVET_VLLM__MODEL` must equal the `--served-model-name` value.
- `TYPEVET_VLLM__TIMEOUT` is in seconds. The default is 300 seconds.
- `TYPEVET_VLLM__API_KEY` must be ASCII. An empty value sends no key.
- `TYPEVET_VLLM__MAX_CONCURRENCY` sets the POST limit for one
  `AsyncVllmGenerationAdapter`. The default is 1.
- `TYPEVET_VLLM__USER_AGENT` replaces the httpx default `User-Agent` header.

`generation_adapter` builds the sync adapter. That adapter does not read
`TYPEVET_VLLM__MAX_CONCURRENCY`. To send parallel requests, use
`async_vllm_generation_adapter`. It builds an `AsyncVllmGenerationAdapter`
with the same variables and uses `TYPEVET_VLLM__MAX_CONCURRENCY` as its limit.
Its HTTP client binds to the first event loop that uses it. Build one adapter
for each event loop, for example inside each `asyncio.run` call. A call on a
different loop raises `RuntimeError("build one adapter per event loop")`
before any request. A retry with the same adapter fails again.

Some proxies block requests that carry a library `User-Agent` header. In that
case, set `TYPEVET_VLLM__USER_AGENT` to a value that the proxy accepts. The
tested run used `curl/8.9.1`.

### Key and network behaviour

- The client sends the key as `Authorization: Bearer <key>`.
- `repr(VllmSettings)` does not show the key.
- Error messages name the variable, never its value.
- The adapters from `generation_adapter` and `async_vllm_generation_adapter`
  mask the key in errors. The port from `open_judgment` also masks it. Each
  error shows `***` in place of the raw or JSON-escaped key. This includes a
  parsed payload. The error has no cause or context.
- An `AsyncVllmGenerationAdapter` that you build yourself does not mask the
  key.
- typevet sets no TLS or proxy options. The httpx defaults apply, so the
  client verifies certificates and reads `HTTPS_PROXY` and the other proxy
  variables.
- The client is the same with or without a key. Only the `Authorization`
  header differs.

Use an HTTPS URL or a local address. A plain HTTP URL sends the key without
encryption.

## Run one typed generation call

```python
from typevet.adapters.inbound import generate, generation_adapter, load_vllm_settings

schema = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["pos", "neg"]}},
    "required": ["sentiment"],
    "additionalProperties": False,
}
settings = load_vllm_settings()
with generation_adapter() as port:
    result = generate(
        port,
        prompt="Classify the review: The blender works well.",
        schema=schema,
        model=settings.model,
    )
print(result.value)
```

The result value matches the schema. The adapter raises a `GenerationError`
subclass when the server rejects the request.

## Run one judgment call

```python
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.domain import Choice

questions = {
    "sentiment": Choice(
        criteria={"pos": "Positive review", "neg": "Negative review"},
        instructions="Classify the review.",
    ),
}
with open_judgment() as session:
    response = session.port.judge(
        "The blender broke after one day.",
        questions,
        session.model,
    )
answer = response.choices["sentiment"]
print(answer.choice, answer.probabilities)
```

Pass `session.model` to `judge`. The session pins that name from
`TYPEVET_VLLM__MODEL`.

## Run the live acceptance test

The live acceptance test runs five sets once and writes one JSON receipt.
Each run costs GPU time. The test skips unless `TYPEVET_REQUIRE_LIVE` is set.

```bash
TYPEVET_REQUIRE_LIVE=1 \
  TYPEVET_BACKEND=vllm \
  TYPEVET_VLLM__BASE_URL=http://127.0.0.1:8000 \
  TYPEVET_VLLM__MODEL=gemma-4-31b-it \
  TYPEVET_VLLM__API_KEY="$VLLM_API_KEY" \
  TYPEVET_VLLM_RECEIPT=scratchpad/vllm/new-receipt.json \
  TYPEVET_VLLM_POD_NOTES='H100 80 GB, vLLM v0.30.0, revision 842da37' \
  uv run pytest evals/tests/live/test_vllm_acceptance_live.py -m live -q -s
```

- `TYPEVET_VLLM_RECEIPT` must name a new file. The test fails when the file
  already exists.
- The nearest existing parent directory must be writable.
- Do not put the key in `TYPEVET_VLLM_POD_NOTES`.
- A missing or invalid variable fails the test before any network call.

The test prints the receipt path and its sha256 digest.

## Limits

- The evidence is one pin and one run.
- Other models are not tested.
- Other vLLM versions are not tested.
- Quantized weights are not tested.
- Mixed scoring batches are not tested. See
  [vllm-project/vllm#51789](https://github.com/vllm-project/vllm/issues/51789).
- The `/metrics` KV-cache metric names are not tested. The tested run did
  not find a KV-cache usage metric.
- The results do not compare backends. The llama.cpp baseline used a
  different quantization.

## Related pages

- [Configuration](../reference/configuration.md)
- [Supported imports](../reference/supported-imports.md)
- [Run Gemma 4 on llama.cpp](run-gemma4-llamacpp.md)
