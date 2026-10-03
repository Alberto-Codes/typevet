"""The five pre-registered set runners for the #170 vLLM acceptance run.

Each runner takes the shared ``RunState`` and its set result mapping from
``typevet_evals.vllm_acceptance.core``. The sets are generation (#168 P10/P11
and the frozen llama.cpp ``SCHEMA``; since #226 the adapter rejects the P10
schema with ``ValueError`` before any POST), PSAI (#180 rev2 visual Choice matrix plus
four text regressions), CORD (text, image-only and combined arms), order (the
CORD combined arm with the label order reversed) and concurrency (8 generation
calls, 4 in parallel, with a KV-cache read while calls are in flight).
Each CORD arm and the order set's CORD rerun send
``CORD_OFF_OPTION_THRESHOLD``. Every scored row in the PSAI, CORD and order
sets records ``off_option_mass`` and ``off_option_flag`` ([#384][i384],
[#409][i409]).
``DEVIATIONS`` records how this wiring differs from the
pre-registered protocol; the receipt keeps it.

Examples:
    ```python
    from typevet_evals.vllm_acceptance.sets import SET_RUNNERS

    assert SET_RUNNERS[0][0] == "generation"
    ```

See Also:
    - [typevet_evals.vllm_acceptance.core][]: caps, gates and receipt
    - [typevet_evals.cord.expense_receipt_requirement][]: CORD arm rows
    - [typevet_evals.datasets.cord_expense][]: CORD expense claim cases
    - [typevet_evals.datasets.psai_vision][]: PSAI screenshot fixtures
    - [typevet_evals.datasets.psai_vision_controls][]: image control matrix

[i170]: https://github.com/Alberto-Codes/typevet/issues/170
[i384]: https://github.com/Alberto-Codes/typevet/issues/384
[i409]: https://github.com/Alberto-Codes/typevet/issues/409
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Final

from typevet.domain.errors import GenerationError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Choice, Question
from typevet.domain.judgment_response import OffOptionReceipt
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.models import GenerationRequest
from typevet_evals.cord.expense_receipt_requirement import judge_cord_expense_arm
from typevet_evals.datasets.cord_expense import (
    LABEL_ORDER,
    ExpenseCase,
    expense_question,
    load_expense_cases,
)
from typevet_evals.datasets.psai_vision import (
    example_image_input,
    load_vision_smoke,
)
from typevet_evals.datasets.psai_vision_controls import (
    annotation_questions,
    annotation_state,
    noul_polarity,
)
from typevet_evals.vllm_acceptance.core import (
    RunState,
    cord_acceptance,
    cord_passed,
    covered,
    kv_cache_usage,
    psai_gates,
)

DEVIATIONS: Final[tuple[str, ...]] = (
    (
        "Identical /tokenize bodies are answered from a per-run memo; "
        "tokenizer_memo_hits counts them."
    ),
    (
        "Concurrency set: 8 calls, 4 in parallel through the sync "
        "generation_adapter in a thread pool, not AsyncVllmGenerationAdapter."
    ),
    (
        "PSAI visibility Choice uses criteria true/false with no description "
        "text; the #180 criteria wording is not recorded."
    ),
    (
        "CORD set runs text_only, image_only and combined arms plus one "
        "image-only omission call (43 calls); no llama.cpp attachment floor."
    ),
    (
        "The supervisor enforces the 60-minute pod cap and the 20-minute "
        "readiness limit; the harness uses its own transport and ignores "
        "proxy environment variables."
    ),
    (
        "PSAI C10 swapped arm uses the repaired #180 donor "
        "cmcc8u6yd00wr1p1yj7aot3ae from c10_repair, not the first pin."
    ),
    (
        "Generation P10 expects ValueError from the typevet schema check with "
        "0 POSTs (#226), not BackendHttpError after 1 POST."
    ),
)
_RECORD_LOCK = threading.Lock()


def record_call(out: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Add one call's seconds and coverage to its set, thread-safely.

    Args:
        out: Set result mapping.
        row: Call row with ``seconds``, ``error`` and optional probabilities.

    Returns:
        ``row`` unchanged.
    """
    with _RECORD_LOCK:
        out.setdefault("seconds", []).append(row["seconds"])
        coverage = out.setdefault("coverage", {"calls": 0, "covered": 0})
        coverage["calls"] += 1
        coverage["covered"] += int(covered(row))
    return row


def failed_row(started: float, exc: Exception) -> dict[str, Any]:
    """Build the row for a call that raised.

    Args:
        started: ``perf_counter`` value at call start.
        exc: The raised error; its text is already key-masked.

    Returns:
        Row with the keys of a scored row: no label, empty probabilities,
        no off-option mass, a ``False`` off-option flag and the error text.
    """
    return _row(started, None, error=f"{type(exc).__name__}: {exc}")


def _row(started: float, receipt: OffOptionReceipt | None, **fields: Any) -> dict:
    receipt = receipt or OffOptionReceipt()
    return {
        "label": None,
        "probabilities": {},
        "tokens_evaluated": None,
        "seconds": round(perf_counter() - started, 3),
        "error": None,
        "off_option_mass": receipt.off_option_mass,
        "off_option_flag": receipt.off_option_flag,
    } | fields


TEXT_PROMPT: Final[str] = (
    "Classify the sentiment of: 'I love this library.' "
    "Return JSON only matching the schema."
)
TEXT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["pos", "neg", "neu"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["sentiment", "confidence"],
    "additionalProperties": False,
}
PSAI_STATE: Final[str] = "Look at the attached screenshot."
IMAGE_ONLY_STATE: Final[str] = (
    "Expense claim for this receipt: the claimed total is not stated. "
    "Decide whether the claim can be checked against the receipt."
)


CONCURRENT_CALLS: Final[int] = 8
PARALLEL: Final[int] = 4
KV_WAIT_SECONDS: Final[float] = 5.0
CORD_OFF_OPTION_THRESHOLD: Final[float] = 0.25


_CORD: Final[str] = "cord/expense_smoke"


def _score(
    run: RunState, out: dict, state: Any, name: str, question: Question, **judge: Any
) -> dict[str, Any]:
    started = perf_counter()
    try:
        response = run.port.judge(state, {name: question}, run.model, **judge)
    except GenerationError as exc:
        return record_call(out, failed_row(started, exc))
    answer = response.answers[name]
    if isinstance(answer, NoulAnswer):
        label = str(noul_polarity(answer.noul)).lower()
        probs = {"true": answer.noul, "false": 1.0 - answer.noul}
    else:
        label, probs = answer.choice, dict(answer.probabilities)
    tokens, receipt = response.usage.input_tokens, response.off_option.get(name)
    fields = {"label": label, "probabilities": probs, "tokens_evaluated": tokens}
    return record_call(out, _row(started, receipt, **fields))


def _generate(run: RunState, out: dict, request: GenerationRequest) -> dict[str, Any]:
    started, before = perf_counter(), run.counter.calls["model"]
    value, error = None, None
    try:
        value = dict(run.generator.generate(request).value)
    except (GenerationError, ValueError) as exc:
        # ValueError: the adapter rejects a malformed schema before any POST.
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(perf_counter() - started, 3)
    posts = run.counter.calls["model"] - before
    return record_call(
        out, {"value": value, "error": error, "seconds": seconds, "posts": posts}
    )


def _generation_set(run: RunState, out: dict[str, Any]) -> None:
    p11 = run.load("vllm/generation_enum_image.json")["request"]
    p10 = run.load("vllm/http400_invalid_schema.json")["request"]
    data = run.fixtures.joinpath("cord", "expense_smoke", "R03.png").read_bytes()
    image = ImageInput(data=data, mime_type="image/png")
    p11_text = p11["messages"][0]["content"][1]["text"]
    asks = (
        (TEXT_PROMPT, TEXT_SCHEMA, ()),
        (f"{MEDIA_MARKER}\n{p11_text}", p11["structured_outputs"]["json"], (image,)),
        (p10["messages"][0]["content"], p10["structured_outputs"]["json"], ()),
    )
    rows = out.setdefault("rows", [])
    for prompt, schema, media in asks:
        request = GenerationRequest(
            prompt=prompt, schema=schema, model=run.model, media=media
        )
        rows.append(_generate(run, out, request))
    invalid = rows[2]
    rejected = str(invalid["error"]).startswith("ValueError")
    valid = rows[0]["error"] is None and rows[1]["error"] is None
    out["passed"] = valid and rejected and invalid["posts"] == 0


def _gold(raw: Any) -> str:
    return str(raw).lower() if isinstance(raw, bool) else str(raw)


def _psai_set(run: RunState, out: dict[str, Any]) -> None:
    corrected = run.load("psai/vision_choice_evidence/corrected_v1.json")
    outcomes = corrected["matrix_outcomes"]
    present = {
        o["case_id"]: o["image_row_id"] for o in outcomes if o["condition"] == "present"
    }
    donors = {
        o["case_id"]: o["swap_donor_id"]
        for o in outcomes
        if o["condition"] == "swapped"
    }
    donors["C10"] = corrected["c10_repair"]["swap_donor"]
    folder = run.fixtures / "psai" / "vision_smoke"
    smoke = load_vision_smoke((folder / "manifest.json").read_text(encoding="utf-8"))
    examples = {e.unique_data_id: e for e in smoke.examples}
    rows = out.setdefault("rows", [])
    for o in outcomes:
        case, condition = o["case_id"], o["condition"]
        image_uid = {"present": present[case], "swapped": donors.get(case)}.get(
            condition
        )
        if condition == "text_only":
            question = annotation_questions()[o["question"]]
            state, name = annotation_state(examples[present[case]]), o["question"]
        else:
            criteria = dict.fromkeys(("true", "false"))
            question = Choice(criteria=criteria, instructions=o["question"])
            state, name = PSAI_STATE, "visible"
        media = ()
        if image_uid is not None:
            example = examples[image_uid]
            media = (example_image_input(example, lambda f: (folder / f).read_bytes()),)
        row = _score(run, out, state, name, question, media=media)
        rows.append(
            {
                "call_id": o["call_id"],
                "case_id": case,
                "condition": condition,
                "gold": _gold(o["gold"]),
                "image_row_id": image_uid,
                **row,
            }
        )
    out.update(psai_gates(rows))


def _cord_cases(run: RunState) -> tuple[ExpenseCase, ...]:
    return load_expense_cases(
        run.fixtures.joinpath(_CORD, "manifest.json").read_text(encoding="utf-8")
    )


def _cord_image(run: RunState, case: ExpenseCase) -> tuple[ImageInput, ...]:
    data = run.fixtures.joinpath(_CORD, case.image_file_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != case.image_sha256:
        msg = f"{case.image_file_name} digest drifted"
        raise ValueError(msg)
    return (ImageInput(data=data, mime_type=case.image_mime_type),)


def _arm(run: RunState, out: dict, state: str, media: tuple, mode: str) -> dict:
    started = perf_counter()
    try:
        row = judge_cord_expense_arm(
            run.port,
            run.model,
            state,
            media,
            application_mode=mode,
            off_option_threshold=CORD_OFF_OPTION_THRESHOLD,
        )
    except GenerationError as exc:
        row = failed_row(started, exc)
    return record_call(out, row)


def _cord_set(run: RunState, out: dict[str, Any]) -> None:
    cases = _cord_cases(run)
    keys = ("claim_id", "receipt_id", "expected_verdict")
    out["cases"] = [{key: getattr(c, key) for key in keys} for c in cases]
    text_only, image_only, combined = (
        out.setdefault(arm, {}) for arm in ("text_only", "image_only", "combined")
    )
    out["image_only_omission"] = _arm(run, out, IMAGE_ONLY_STATE, (), "text_only")
    for case in cases:
        media = _cord_image(run, case)
        statement = case.model_inputs()["statement"]
        text_only[case.claim_id] = _arm(run, out, statement, (), "text_only")
        if case.receipt_id not in image_only:
            image_only[case.receipt_id] = _arm(
                run, out, IMAGE_ONLY_STATE, media, "image_only"
            )
        combined[case.claim_id] = _arm(run, out, statement, media, "combined")
    out.update(cord_acceptance(out["cases"], combined))
    out["passed"] = cord_passed(out)


def _order_set(run: RunState, out: dict[str, Any]) -> None:
    base = expense_question()
    reversed_order = tuple(reversed(LABEL_ORDER))
    question = Choice(
        criteria={label: base.criteria[label] for label in reversed_order},
        instructions=base.instructions,
    )
    cord = run.sets["cord"]
    out.update(label_order=list(reversed_order), record_only=True)
    combined = out.setdefault("combined", {})
    guard = {"off_option_threshold": CORD_OFF_OPTION_THRESHOLD}
    for case in _cord_cases(run):
        statement = case.model_inputs()["statement"]
        media = _cord_image(run, case)
        combined[case.claim_id] = _score(
            run, out, statement, "expense", question, media=media, **guard
        )
    out.update(cord_acceptance(cord["cases"], combined))
    out["flips"] = sum(
        row["label"] != cord["combined"][claim_id]["label"]
        for claim_id, row in combined.items()
    )


def _concurrency_set(run: RunState, out: dict[str, Any]) -> None:
    """Send 8 generation calls, 4 in parallel, and read the KV cache in flight.

    The ``/metrics`` read waits until ``PARALLEL`` requests entered the
    counting transport, or ``KV_WAIT_SECONDS`` passed; it reads on timeout.

    Args:
        run: Shared run state.
        out: Concurrency set result mapping.
    """
    request = GenerationRequest(prompt=TEXT_PROMPT, schema=TEXT_SCHEMA, model=run.model)
    started, target = perf_counter(), run.counter.calls["model"] + PARALLEL
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = [
            pool.submit(_generate, run, out, request) for _ in range(CONCURRENT_CALLS)
        ]
        run.counter.wait_for("model", target, KV_WAIT_SECONDS)
        out["kv_cache_in_flight"] = kv_cache_usage(run.client)
        rows = [future.result() for future in futures]
    out.update(
        rows=rows,
        parallel=PARALLEL,
        record_only=True,
        wall_seconds=round(perf_counter() - started, 3),
        errors=sum(row["error"] is not None for row in rows),
    )


SET_RUNNERS: Final[
    tuple[tuple[str, Callable[[RunState, dict[str, Any]], None]], ...]
] = (
    ("generation", _generation_set),
    ("psai", _psai_set),
    ("cord", _cord_set),
    ("order", _order_set),
    ("concurrency", _concurrency_set),
)
