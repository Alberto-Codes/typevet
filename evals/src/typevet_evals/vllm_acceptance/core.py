"""Call caps, receipt assembly and gates for the #170 vLLM acceptance run.

``run_acceptance`` runs set runners through the public entry points
``generation_adapter`` and ``open_judgment`` with ``TYPEVET_BACKEND=vllm``.
The runners live in ``typevet_evals.vllm_acceptance.sets``, which imports
this module; this module never imports it. Every request passes through
``CountingTransport``, which counts model, tokenizer and metadata calls and
raises ``CallCapReached`` before a request that would pass a cap. The run then
stops and still returns a receipt. The harness makes no transport retry. The
receipt keeps the CORD ``cases`` and ``combined`` rows at the top level, so
``typevet_evals.cli.cord_semantic_acceptance`` reads the receipt file
directly. The call caps, ``CountingTransport``, ``CallCapReached`` and
``kv_cache_usage`` live in ``typevet_evals.vllm_acceptance.transport``; this
module re-exports all of them except ``CallCapReached`` (#229). The
receipt and its error message never hold the key or an extra header value
(#348). That includes a header value that a server body uses as a key (#351).

Examples:
    ```python
    import httpx

    from typevet_evals.vllm_acceptance.core import run_acceptance, write_receipt
    from typevet_evals.vllm_acceptance.sets import DEVIATIONS, SET_RUNNERS

    receipt = run_acceptance(
        env,
        inputs,
        transport=httpx.HTTPTransport(),
        runners=SET_RUNNERS,
        deviations=DEVIATIONS,
    )
    write_receipt(path, receipt, settings=load_vllm_settings(env))
    ```

See Also:
    - [typevet_evals.vllm_acceptance.sets][]: the five set runners
    - [typevet_evals.vllm_acceptance.transport][]: call caps and
      ``/metrics`` read
    - [typevet.adapters.inbound.backend_settings][]: backend selection
    - [typevet_evals.cord.semantic_acceptance][]: CORD floors
    - [typevet_evals.runner.live_gate][]: require-live switch
    - [typevet_evals.experiment_identity][]: baseline commit reader

[i170]: https://github.com/Alberto-Codes/typevet/issues/170
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from typevet.adapters.diagnostics.redaction import REDACTED
from typevet.adapters.inbound.backend_settings import (
    VllmSettings,
    generation_adapter,
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet_evals.cord.semantic_acceptance import accept_combined_receipt
from typevet_evals.experiment_identity import read_baseline_commit
from typevet_evals.runner.live_gate import require_live_enabled
from typevet_evals.vllm_acceptance.transport import (
    AcceptanceStoppedError,
    CallCaps,
    CountingTransport,
    kv_cache_usage,
)

ATTACHMENT_DELTA_FLOOR: Final[int] = 50
SWAP_MARGIN_FLOOR: Final[float] = 0.10
PSAI_CASES: Final[int] = 4
SWAP_PASS_FLOOR: Final[int] = 3
_DENIED: Final[frozenset[int | None]] = frozenset({401, 403})
_NOT_VLLM: Final = "TYPEVET_BACKEND must be vllm for the vLLM acceptance run"
_GATED: Final[tuple[str, ...]] = ("generation", "psai", "cord")
_BOUND: Final = "(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"
_ECHOES: Final = (("pins", "version"), ("pins", "served_models"), ("error", "message"))
OMITTED_RULE: Final[str] = (
    "An omitted row counts in omitted_credited when its answer equals the "
    "gold key; no omitted row is added to the present, swapped or text counts."
)


@dataclass(frozen=True, slots=True)
class AcceptanceInputs:
    """Paths the harness reads.

    Attributes:
        fixtures_root (Path): The ``tests/fixtures`` directory.
        repo_root (Path): Checkout root for the source SHA.

    Examples:
        ```python
        AcceptanceInputs(Path("tests/fixtures"), Path("."))
        ```
    """

    fixtures_root: Path
    repo_root: Path


@dataclass(slots=True)
class RunState:
    """State the set runners share during one acceptance run.

    Attributes:
        port (Any): Key-masking ``JudgmentPort`` from ``open_judgment``.
        generator (Any): Adapter from ``generation_adapter``.
        model (str): Served model name.
        fixtures (Path): The ``tests/fixtures`` directory.
        counter (CountingTransport): Shared call counter.
        sets (dict[str, dict[str, Any]]): Per-set results in run order.
        client (httpx.Client): Session client for ``/metrics`` reads.

    Examples:
        ```python
        RunState(port, generator, "served-model", fixtures, counter, {}, client)
        ```
    """

    port: Any
    generator: Any
    model: str
    fixtures: Path
    counter: CountingTransport
    sets: dict[str, dict[str, Any]]
    client: httpx.Client

    def load(self, relative: str) -> Any:
        """Parse one JSON fixture.

        Args:
            relative: Path under ``fixtures``.

        Returns:
            The parsed JSON value.
        """
        return json.loads((self.fixtures / relative).read_text(encoding="utf-8"))


def latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    """Return the sample count and nearest-rank p50 and p95 in seconds.

    Args:
        samples: Per-call seconds.

    Returns:
        ``n``, ``p50`` and ``p95``; the percentiles are ``None`` when empty.
    """
    ordered = sorted(samples)
    return {"n": len(ordered), "p50": _rank(ordered, 0.5), "p95": _rank(ordered, 0.95)}


def _rank(ordered: list[float], quantile: float) -> float | None:
    if not ordered:
        return None
    return ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)]


def covered(row: Mapping[str, Any]) -> bool:
    """Return whether one call answered with finite probabilities.

    Args:
        row: Call row with ``error`` and optional ``probabilities``.

    Returns:
        ``True`` when there is no error and every probability is finite.
    """
    probs = row.get("probabilities") or {}
    finite = all(isinstance(p, float) and math.isfinite(p) for p in probs.values())
    return row.get("error") is None and finite


def _present_ok(present: Mapping[str, Any], omitted: Mapping[str, Any] | None) -> bool:
    if omitted is None or present["label"] != present["gold"]:
        return False
    tokens = (present["tokens_evaluated"], omitted["tokens_evaluated"])
    if not all(isinstance(t, int) for t in tokens):
        return False
    return tokens[0] - tokens[1] >= ATTACHMENT_DELTA_FLOOR


def _swap_ok(present: Mapping[str, Any], swapped: Mapping[str, Any] | None) -> bool:
    """Return whether the swapped image moved the answer away from gold.

    The #180 gates say a swap counts when the answer "moves away from gold".
    "Away" assumes that the present answer started at gold. This rule is
    stricter than that text: the label change counts only when the present
    label equals gold. A wrong present answer that stays wrong after the
    swap counts only through the probability margin. The rule can only lower
    ``swapped``, and ``passed`` already needs present 4/4 (#216).

    Args:
        present: Scored ``present`` row with ``gold``, ``label`` and
            ``probabilities``.
        swapped: Scored ``swapped`` row, or ``None`` when it is missing.

    Returns:
        ``True`` when the gold probability drops by at least
        ``SWAP_MARGIN_FLOOR``, or when a correct present label changes to a
        label that is not gold.
    """
    if swapped is None or swapped["label"] is None:
        return False
    gold = present["gold"]
    before, after = present["probabilities"], swapped["probabilities"]
    margin = before.get(gold, 0.0) - after.get(gold, 0.0)
    moved = present["label"] == gold and swapped["label"] != gold
    return margin >= SWAP_MARGIN_FLOOR or moved


def psai_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the #180 rev2 gates to scored PSAI rows.

    Present passes when it predicts the gold key and the image adds at least
    ``ATTACHMENT_DELTA_FLOOR`` prompt tokens over the omitted arm. Swapped
    passes when P(gold) drops by ``SWAP_MARGIN_FLOOR`` or the label leaves a
    correct present answer. Omitted arms are never credited. Text rows must
    match gold.

    Args:
        rows: Rows with ``case_id``, ``condition``, ``gold``, ``label``,
            ``probabilities``, ``tokens_evaluated`` and ``error``.

    Returns:
        Pass counts, ``omitted_credited`` under ``OMITTED_RULE`` (recorded,
        not gated), ``covered`` and ``passed`` (present 4/4, swapped at least
        3/4, text 4/4 and every call covered).
    """
    by = {(r["case_id"], r["condition"]): r for r in rows}
    cases = [case for case, condition in by if condition == "present"]
    text = [r for r in rows if r["condition"] == "text_only"]
    present = sum(
        _present_ok(by[(c, "present")], by.get((c, "omitted"))) for c in cases
    )
    swapped = sum(_swap_ok(by[(c, "present")], by.get((c, "swapped"))) for c in cases)
    text_ok = sum(r["label"] == r["gold"] for r in text)
    omitted = [r for r in rows if r["condition"] == "omitted"]
    covered_all = all(map(covered, rows))
    passed = (
        covered_all
        and present == len(cases) == PSAI_CASES
        and swapped >= SWAP_PASS_FLOOR
        and text_ok == len(text) == PSAI_CASES
    )
    return {
        "present": present,
        "swapped": swapped,
        "text": text_ok,
        "omitted_credited": sum(r["label"] == r["gold"] for r in omitted),
        "omitted_rule": OMITTED_RULE,
        "covered": covered_all,
        "passed": passed,
    }


def cord_passed(out: Mapping[str, Any]) -> bool:
    """Return whether the CORD set passed: accepted and every call covered.

    Args:
        out: CORD set mapping with ``accepted`` and ``coverage``.

    Returns:
        ``True`` only when both hold.
    """
    cov = out.get("coverage") or {}
    return out.get("accepted") is True and cov.get("calls") == cov.get("covered")


def cord_acceptance(cases: list, combined: dict) -> dict[str, Any]:
    """Apply the CORD #161 floors to combined rows.

    Args:
        cases: Case rows with ``claim_id`` and ``expected_verdict``.
        combined: Combined rows keyed by claim id.

    Returns:
        ``accepted``, per-check rows and failure reasons. A malformed row set
        is not accepted.
    """
    try:
        outcome = accept_combined_receipt({"cases": cases, "combined": combined})
    except ValueError as exc:
        return {"accepted": False, "checks": [], "failures": [str(exc)]}
    checks = [
        {"name": c.name, "limit": c.limit, "measured": c.measured, "passed": c.passed}
        for c in outcome.checks
    ]
    failures = list(outcome.failures)
    return {"accepted": outcome.accepted, "checks": checks, "failures": failures}


def _get(client: httpx.Client, path: str) -> dict[str, Any]:
    try:
        response = client.get(path)
    except httpx.HTTPError as exc:
        return {"status": None, "body": None, "error": type(exc).__name__}
    try:
        body = response.json()
    except ValueError:
        body = response.text[:200]
    return {"status": response.status_code, "body": body}


def _pins(
    client: httpx.Client, environ: Mapping[str, str], inputs: AcceptanceInputs
) -> dict[str, Any]:
    settings = load_vllm_settings(environ)
    version, models = _get(client, "/version"), _get(client, "/v1/models")
    body = models["body"]
    data = body.get("data") if isinstance(body, dict) else None
    rows = [m for m in data or [] if isinstance(m, dict)]
    served = [{"id": m.get("id"), "root": m.get("root")} for m in rows]
    return {
        "version": version,
        "models_status": models["status"],
        "served_models": served,
        "configured_model": settings.model,
        "base_url": settings.base_url,
        "user_agent": settings.user_agent,
        "source_sha": read_baseline_commit(inputs.repo_root),
        "pod_notes": environ.get("TYPEVET_VLLM_POD_NOTES", "").strip() or "unknown",
        "kv_cache_usage": "unknown",
    }


def _preflight(pins: Mapping[str, Any]) -> None:
    statuses = {pins["version"]["status"], pins["models_status"]}
    reason = None
    if statuses <= _DENIED:
        reason = "preflight denied: /version and /v1/models returned 401 or 403"
    elif pins["version"]["status"] != httpx.codes.OK:
        reason = f"preflight: /version returned {pins['version']['status']}"
    elif pins["models_status"] != httpx.codes.OK:
        reason = f"preflight: /v1/models returned {pins['models_status']}"
    elif pins["configured_model"] not in {m["id"] for m in pins["served_models"]}:
        reason = "preflight: /v1/models does not list TYPEVET_VLLM__MODEL"
    if reason is not None:
        raise AcceptanceStoppedError(reason)


def run_acceptance(
    environ: Mapping[str, str],
    inputs: AcceptanceInputs,
    *,
    transport: httpx.BaseTransport,
    runners: Sequence[tuple[str, Callable[[RunState, dict[str, Any]], None]]],
    deviations: Sequence[str] = (),
    caps: CallCaps | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Run the five sets once and return the receipt mapping.

    Args:
        environ: Mapping with ``TYPEVET_BACKEND=vllm`` and ``TYPEVET_VLLM__*``.
        inputs: Fixture and checkout paths.
        transport: Transport that sends requests; the caller closes it.
        runners: Named set runners, run in order.
        deviations: Recorded differences from the pre-registered protocol.
        caps: Call limits. Defaults to ``CallCaps()``.
        receipt_path: When set, the receipt is written there with
            ``write_receipt`` and the loaded settings, even when the run
            raises.

    Returns:
        Receipt with ``pins``, ``sets``, ``calls``, ``coverage``, ``stopped``,
        ``passed`` and the CORD ``cases`` and ``combined`` rows. A stop rule
        leaves ``stopped`` set and ``passed`` false. Any other error is
        recorded as ``error`` (type, and message with the key and header
        values masked), the receipt is
        written, and the error is raised again.

    Raises:
        ValueError: When the backend is not ``vllm`` or a setting is invalid.
    """
    if load_backend(environ) != "vllm":
        raise ValueError(_NOT_VLLM)
    settings = load_vllm_settings(environ)
    counter = CountingTransport(transport, caps or CallCaps())
    receipt: dict[str, Any] = {"issue": 170, "artifact": "vllm_acceptance_v1"}
    receipt.update(caps=asdict(counter.caps), transport_retries=0, stopped=None)
    receipt.update(pins={}, sets={}, deviations=list(deviations))
    try:
        with (
            generation_adapter(environ, transport=counter) as generator,
            open_judgment(environ, transport=counter) as session,
        ):
            receipt["pins"] = _pins(session.client, environ, inputs)
            _preflight(receipt["pins"])
            state = (settings.model, inputs.fixtures_root, counter, receipt["sets"])
            run = RunState(session.port, generator, *state, session.client)
            for name, runner in runners:
                runner(run, run.sets.setdefault(name, {}))
            receipt["pins"]["kv_cache_usage"] = kv_cache_usage(session.client)
    except AcceptanceStoppedError as exc:
        receipt["stopped"] = str(exc)
    except Exception as exc:
        message = _masked(_masked_headers(str(exc), settings), settings)
        receipt["stopped"] = f"error: {type(exc).__name__}"
        receipt["error"] = {"type": type(exc).__name__, "message": message}
        raise
    finally:
        _finish(receipt, counter)
        if receipt_path is not None:
            write_receipt(receipt_path, receipt, settings=settings)
    return receipt


def _finish(receipt: dict[str, Any], counter: CountingTransport) -> dict[str, Any]:
    sets = receipt["sets"]
    for out in sets.values():
        out["latency"] = latency_summary(out.pop("seconds", []))
    cord = sets.get("cord", {})
    gated = all(sets.get(name, {}).get("passed") is True for name in _GATED)
    receipt.update(
        calls=dict(counter.calls),
        tokenizer_memo_hits=counter.tokenizer_memo_hits,
        coverage={name: out.get("coverage") for name, out in sets.items()},
        cases=cord.get("cases", []),
        combined=cord.get("combined", {}),
        passed=receipt["stopped"] is None and gated,
    )
    return receipt


def _key_needle(settings: VllmSettings, *, escaped: bool) -> str:
    """Return the configured key, raw or JSON-escaped, without binding it.

    Returns:
        The key text to replace with ``REDACTED``.
    """
    return json.dumps(settings.api_key)[1:-1] if escaped else str(settings.api_key)


def _masked_headers(value: Any, settings: VllmSettings | None) -> Any:
    """Mask each extra header value as a whole token in the strings of ``value``.

    A gateway can echo a request header into a body or an error that the
    receipt records (#348). A value matches only between characters that are
    not ASCII letters or digits, as in the adapter errors, so a value such as
    ``1`` leaves ``HTTP 401`` readable. Strings and the keys of nested dicts
    are masked. The keys of the outer dict, and numbers, keep their form (#351).

    Args:
        value: String, or a dict or list of JSON values.
        settings: vLLM settings with the extra headers, or ``None``.

    Returns:
        ``value`` with each header value replaced by ``REDACTED``.
    """
    headers = {} if settings is None else settings.headers
    values = sorted(filter(None, headers.values()), key=len, reverse=True)
    patterns = [re.compile(_BOUND.format(re.escape(v))) for v in values]
    return _masked_tokens(value, patterns) if patterns else value


def _masked_tokens(
    value: Any, patterns: list[re.Pattern[str]], *, keys: bool = False
) -> Any:
    if isinstance(value, str):
        for pattern in patterns:
            value = pattern.sub(REDACTED, value)
        return value
    if isinstance(value, Mapping):
        names = _masked_keys(list(value), patterns) if keys else list(value)
        items = zip(names, value.values(), strict=True)
        return {n: _masked_tokens(v, patterns, keys=True) for n, v in items}
    if isinstance(value, list | tuple):
        return [_masked_tokens(item, patterns, keys=keys) for item in value]
    return value


def _masked_keys(names: list[Any], patterns: list[re.Pattern[str]]) -> list[Any]:
    """Mask string keys as whole tokens, and keep them distinct (#351).

    A key that the masking changes gets the first free name of
    ``***``, ``***_2``, ``***_3`` and so on, in key order. A key that the
    masking does not change keeps its name. Thus no key collapses.

    Returns:
        The masked keys, in the same order as ``names``.
    """
    masked = [_masked_tokens(n, patterns) if isinstance(n, str) else n for n in names]
    taken = {new for old, new in zip(names, masked, strict=True) if old == new}
    out = []
    for old, new in zip(names, masked, strict=True):
        name, count = new, 1
        while old != new and name in taken:
            count += 1
            name = f"{new}_{count}"
        taken.add(name)
        out.append(name)
    return out


def _masked_echoes(
    receipt: Mapping[str, Any], settings: VllmSettings | None
) -> dict[str, Any]:
    """Mask header values only in the receipt parts that a gateway can echo.

    Those parts are ``pins.version`` and ``pins.served_models``, read from the
    server, and ``error.message``. The keys and the values that typevet sets,
    such as ``stopped``, ``passed`` and ``pins.configured_model``, stay as
    they are, so a header value equal to one of them cannot change the
    receipt schema (#348). The ``status`` and ``body`` keys of
    ``pins.version`` and the ``id`` and ``root`` keys of each served model
    are typevet keys too. Keys below them come from the server and are
    masked (#351).

    Args:
        receipt: Mapping from ``run_acceptance``.
        settings: vLLM settings with the extra headers, or ``None``.

    Returns:
        A copy of ``receipt`` with those parts masked.
    """
    masked = dict(receipt)
    for parent, child in _ECHOES:
        section = masked.get(parent)
        if isinstance(section, Mapping) and child in section:
            echo = _masked_headers(section[child], settings)
            masked[parent] = {**section, child: echo}
    return masked


def _masked(text: str, settings: VllmSettings | None) -> str:
    if settings is None or not settings.api_key:
        return text
    for escaped in (False, True):
        text = text.replace(_key_needle(settings, escaped=escaped), REDACTED)
    return text


def write_receipt(
    path: Path, receipt: Mapping[str, Any], *, settings: VllmSettings | None
) -> str:
    """Write the receipt JSON once, with secrets masked, and return its sha256.

    Each extra header value is masked as a whole token only in the parts that
    ``_masked_echoes`` names, before the dump, so typevet keys, numbers and
    the values that typevet sets keep their form (#348, #351). The key is then masked
    anywhere in the JSON text. The key is read from ``settings`` only inside
    the masking step, so no frame of this function binds the raw key to a
    local. ``VllmSettings`` leaves the key out of its
    ``repr``.

    Args:
        path: New file path; parent directories are created.
        receipt: Mapping from ``run_acceptance``.
        settings: vLLM settings whose ``api_key`` is masked, raw and
            JSON-escaped, and whose ``headers`` values are masked, or
            ``None`` for no masking.

    Returns:
        Hex sha256 of the written bytes.

    Raises:
        FileExistsError: When ``path`` already exists.
    """
    masked = _masked_echoes(receipt, settings)
    text = _masked(json.dumps(masked, indent=2) + "\n", settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def live_gate_reason(environ: Mapping[str, str]) -> str | None:
    """Return why the live acceptance cannot run, or ``None`` when it can.

    Args:
        environ: Mapping to read for vLLM settings and the receipt path.

    Returns:
        A reason when ``TYPEVET_REQUIRE_LIVE`` is not truthy, a vLLM setting
        is missing or invalid, the backend is not ``vllm`` or
        ``TYPEVET_VLLM_RECEIPT`` is empty, already exists or has no writable
        existing parent directory. Names variables, never values.
    """
    if not require_live_enabled():
        return "set TYPEVET_REQUIRE_LIVE=1 to run the paid vLLM acceptance"
    try:
        backend = load_backend(environ)
        load_vllm_settings(environ)
    except ValueError as exc:
        return str(exc)
    if backend != "vllm":
        return _NOT_VLLM
    return _receipt_path_reason(environ.get("TYPEVET_VLLM_RECEIPT", "").strip())


def _receipt_path_reason(raw: str) -> str | None:
    if not raw:
        return "TYPEVET_VLLM_RECEIPT must name the receipt path"
    path = Path(raw).absolute()
    parent = next(p for p in path.parents if p.exists())
    if path.exists():
        return "TYPEVET_VLLM_RECEIPT names a file that already exists"
    if not (parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)):
        return "TYPEVET_VLLM_RECEIPT parent directory is not writable"
    return None
