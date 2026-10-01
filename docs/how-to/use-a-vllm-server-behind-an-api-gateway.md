---
status: draft
---

# Use a vLLM server behind an API gateway

Kind: how-to.

Status: **draft**.

Use this page when an API gateway is in front of your vLLM server. Apigee,
Kong and Azure API Management are examples. The gateway can use a path prefix, its own key
header and extra headers. typevet sends them on each vLLM call.
[Issue #331](https://github.com/Alberto-Codes/typevet/issues/331) holds the contract.

No live gateway call proves these steps. Contract tests on `httpx.MockTransport`
prove them.

## Before you start

- Start a vLLM server. See [Serve typevet on vLLM](serve-typevet-on-vllm.md).
- Get the gateway URL, the key header name and each required extra header.
- Use an HTTPS gateway URL.

## Set the gateway variables

1. Set the backend and the gateway URL with its path prefix.

    ```bash
    export TYPEVET_BACKEND=vllm
    export TYPEVET_VLLM__BASE_URL=https://gw.example.com/vllm
    export TYPEVET_VLLM__MODEL=gemma-4-31b-it
    ```

2. Set the gateway key.

    ```bash
    export TYPEVET_VLLM__API_KEY='<gateway key>'
    ```

3. Set the header that carries the key.
    This example uses `X-API-Key`.

    ```bash
    export TYPEVET_VLLM__AUTH_HEADER=X-API-Key
    ```

4. Set the scheme before the key.
    An empty value sends the key with no scheme.

    ```bash
    export TYPEVET_VLLM__AUTH_SCHEME=
    ```

5. Set the extra headers as one JSON object of string values.

    ```bash
    export TYPEVET_VLLM__HEADERS='{"X-Tenant":"acme"}'
    ```

6. Optional: set a request-id header.
    typevet then sends a new UUID4 hex value in it on each request.

    ```bash
    export TYPEVET_VLLM__REQUEST_ID_HEADER=X-Request-Id
    ```

## Check what typevet sends

With the variables above, each vLLM call goes to the gateway prefix:

| Call | URL |
|---|---|
| Generation, sync and async | `https://gw.example.com/vllm/v1/chat/completions` |
| Judgment scoring | `https://gw.example.com/vllm/v1/chat/completions` |
| Judgment tokenization | `https://gw.example.com/vllm/tokenize` |

A trailing slash on `TYPEVET_VLLM__BASE_URL` gives the same URLs.

Each request carries these headers:

- `X-API-Key: <gateway key>`, and no `Authorization` header.
- `X-Tenant: acme`.
- `X-Request-Id: <32 hex characters>`, when you set step 6.

These entry points build their clients from the same settings:
`generation_adapter`, `async_vllm_generation_adapter` and `open_judgment`.
The judgevet bridge and the served-template probe use `open_judgment`.
Thus they send the same headers.

Set no gateway variable to keep the direct behaviour.
typevet then sends `Authorization: Bearer <key>` as before.

## Header rules

typevet checks the headers when it builds `VllmSettings`.
A failed check raises `ValueError`.
The message names the variable and never shows a value.

| Rule | Limit |
|---|---|
| Header name | An HTTP token of at most 128 bytes |
| Header value | ASCII characters 0x20 to 0x7E only, at most 2,048 bytes |
| Number of extra headers | At most 32 |
| Names and values together | At most 8,192 bytes |

typevet refuses these names in `TYPEVET_VLLM__HEADERS`, in any letter case:

- The hop-by-hop headers: `Connection`, `Keep-Alive`, `Proxy-Authenticate`,
  `Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding` and `Upgrade`.
- `Host` and `Content-Length`.
- `Content-Type`, `Accept`, `Accept-Encoding` and `User-Agent`. typevet sets
  them. Only `TYPEVET_VLLM__USER_AGENT` changes the agent string.
- The header in `TYPEVET_VLLM__AUTH_HEADER`. Set the key in
  `TYPEVET_VLLM__API_KEY` instead.

typevet sends each value as written.
It does not expand `$NAME` and does not run `!command`.

## Handle gateway errors

- A gateway 429 raises `BackendHttpError` with `status_code` 429.
  Your code decides whether to retry. typevet adds no retry.
- Read `retry_after_seconds` for the wait that `Retry-After` asks for.
  It is `None` when the gateway sends no valid value.
- Read `rate_limit` for the `x-ratelimit-*` and `ratelimit-*` headers.
  It never holds the auth header or a `TYPEVET_VLLM__HEADERS` name.
- An HTML error body from the gateway is not in the error.
  `body_snippet` is empty.
- typevet follows no redirect.
  A 3xx status raises `BackendHttpError` with that `status_code`.
  The error does not show the `Location` header.
- Each error from these entry points masks the key and each value in
  `TYPEVET_VLLM__HEADERS` with `***`.
  The masked error has no cause or context.

[Security](../reference/security.md#api-keys) lists the limits of this masking.
[Retry hints](../reference/errors.md#retry-hints-on-a-vllm-error) gives the `Retry-After` rules.

## Correlate a gateway log line

Do this step when a gateway log line must match a typevet result or error.
[Issue #356](https://github.com/Alberto-Codes/typevet/issues/356) holds the contract.

1. Set `TYPEVET_VLLM__REQUEST_ID_HEADER` as in step 6.
    Configure the gateway to log that header.

2. Read the id of each judgment question from the response.

    ```python
    from typevet.adapters.inbound.backend_settings import open_judgment

    with open_judgment() as session:
        response = session.port.judge("state", questions, session.model)
    for name, request_id in response.request_ids.items():
        print(name, request_id)
    ```

    The id is the value in the scoring request of that question.
    When a scoring wrapper sends more than one request for a question, the last id is kept.

3. Read `request_id` on a `BackendHttpError` or a `TransportError`.
    It is the id of the request that failed.
    This works for `generation_adapter`, `async_vllm_generation_adapter` and `open_judgment`.

4. Search the gateway log for that id.

Without `TYPEVET_VLLM__REQUEST_ID_HEADER`, `request_ids` is empty and `request_id` is `None`.
A successful generation result does not carry the id.
The `/tokenize` requests of a judgment are not in `request_ids`.
typevet never logs the id. The id is not a secret, so typevet does not mask it.

## Related pages

- [Configuration](../reference/configuration.md#vllm-server)
- [Serve typevet on vLLM](serve-typevet-on-vllm.md)
- [Security](../reference/security.md)
- [Errors](../reference/errors.md)
