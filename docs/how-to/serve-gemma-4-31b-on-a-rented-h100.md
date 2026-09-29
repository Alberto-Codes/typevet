# Serve Gemma 4 31B on a rented H100

Kind: how-to.

Use this page to rent one H100 on RunPod and serve Gemma 4 31B BF16 with stock vLLM.
Then point typevet at the server and run one judgment.
The steps follow the pod of the
[#236 attempt-2 live receipt](https://github.com/Alberto-Codes/typevet/issues/236#issuecomment-5898660690).
[Performance on one H100](../reference/performance.md) gives the measured throughput of that pod.

[Serve typevet on vLLM](serve-typevet-on-vllm.md) states the tested pin, each setting and the key behaviour.
This page does not repeat them.

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with an API key.
- A budget of about $3.49 per hour for the pod.
- A typevet checkout with `uv sync` complete.
- Acceptance of the Gemma 4 licence. The weights are not gated. A Hugging Face token is optional.

## Step 1: Create the pod

Create the pod from the RunPod console or the RunPod API.
The contract and receipt record the image, flags, disk and GPU.
The console settings below (port, start command, entrypoint) are the recommended setup, not recorded values:

| Parameter | Value |
|---|---|
| GPU | `NVIDIA H100 80GB HBM3` (H100 SXM), 1 GPU |
| Cloud type | Secure |
| Data center | Any with H100 SXM stock. The measured pod ran in US-MO-1 |
| Image | `vllm/vllm-openai:v0.30.0`, default entrypoint |
| Disk | 150 GB |
| Exposed HTTP port | `8000` |
| Environment | `VLLM_API_KEY=<your-key>`, optional `HF_TOKEN=<your-hugging-face-token>` |

Set the container start command to these server arguments:

```text
--model google/gemma-4-31B-it
--revision 842da3794eaa0b77d5f08bae87a17459d91ff475
--max-model-len 8192
--gpu-memory-utilization 0.95
--max-num-seqs 64
--enable-prefix-caching
--logprobs-mode raw_logprobs
--served-model-name gemma-4-31b-it
```

`--max-num-seqs 64` lets vLLM run up to 64 requests in one batch.
`--enable-prefix-caching` lets vLLM reuse the cache for a repeated prompt prefix.
The #170 pin used `--max-num-seqs 4` and image limits.
This pod used neither the image limit nor the lower sequence count.

Use a new random value for `VLLM_API_KEY`.
Do not put the key in a file in the repository.

## Step 2: Wait for the model

The RunPod proxy gives the pod an HTTPS address of this form:

```text
https://<pod-id>-8000.proxy.runpod.net
```

Poll `/v1/models` until it answers.
The proxy blocks some library `User-Agent` headers, so send `curl/8.9.1`:

```bash
export POD_URL='https://<pod-id>-8000.proxy.runpod.net'
export VLLM_API_KEY='<your-key>'
until curl -sf -A 'curl/8.9.1' -H "Authorization: Bearer $VLLM_API_KEY" \
    "$POD_URL/v1/models" > /dev/null; do
  sleep 20
done
curl -s -A 'curl/8.9.1' -H "Authorization: Bearer $VLLM_API_KEY" "$POD_URL/version"
```

Expect `{"version":"0.30.0"}`.
The measured cold start was 6 min 46 s, from pod creation to the first `/v1/models` answer.

An earlier pod in US-NE-1 returned 404 for 21 min and never served the model.
If `/v1/models` does not answer after about 15 minutes, delete the pod.
This limit is a recommendation. A failed pod on the same day stayed silent for 21 minutes.
Then create a new pod in a different data center.

## Step 3: Point typevet at the pod

```bash
export TYPEVET_BACKEND=vllm
export TYPEVET_VLLM__BASE_URL="$POD_URL"
export TYPEVET_VLLM__MODEL=gemma-4-31b-it
export TYPEVET_VLLM__API_KEY="$VLLM_API_KEY"
export TYPEVET_VLLM__USER_AGENT=curl/8.9.1
```

`TYPEVET_VLLM__MODEL` must equal the `--served-model-name` value.
[Configuration](../reference/configuration.md) lists each `TYPEVET_VLLM__*` variable.

## Step 4: Run one judgment

```bash
uv run python -c "
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.domain import Noul
q = {'is_scam': Noul(instructions='Is this message a scam?',
     criteria={'true': 'A scam', 'false': 'Not a scam'})}
with open_judgment() as session:
    r = session.port.judge('Your parcel is on its way.', q, session.model)
print('noul', r.nouls['is_scam'].noul)
"
```

The command prints the probability of yes, a value from 0 to 1.
The value for this text is not recorded in a receipt.

For parallel records, run several `judge` calls from a thread pool.
The measured run used a thread pool of up to 64 workers.

## Step 5: Delete the pod

An idle pod bills at the full rate.
Delete the pod from the RunPod console or the RunPod API when you finish.
Then list your pods and confirm that the pod is gone.

## Cost

| Item | Value |
|---|---|
| Pod price | $3.49 per hour |
| Measured pod life | 20:41:39Z to 20:54:27Z, about 12 min 48 s |
| Measured pod cost | About $0.75 |
| Cold start share of that life | 6 min 46 s |

The pod that never served cost about $1.29 before its deletion.
The [supervisor pod record](https://github.com/Alberto-Codes/typevet/issues/236#issuecomment-5898685257) gives both costs.

## Limits

- The evidence is one run on one pod with one pin.
- Other GPUs, models, precisions and vLLM versions are not tested.
- The measured client ran through the RunPod proxy, so its latency includes the proxy.
- The measured texts were short (89 to 254 mean prompt tokens); throughput does not transfer to longer prompts.
- Calibration differs by set: DIFrauD SMS failed parity. See [Performance on one H100](../reference/performance.md).

## Related pages

- [Performance on one H100](../reference/performance.md)
- [Serve typevet on vLLM](serve-typevet-on-vllm.md)
- [Configuration](../reference/configuration.md)
- [Typed-judgment release support matrix](../reference/typed-judgment-release-support-matrix.md)
