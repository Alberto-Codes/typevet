"""Terminal demo: typed judgments on one customer message and two receipt images.

Scenario: a customer reports a duplicate coffee charge. Three typed questions
(Noul, Choice and Score) judge the message. Next, one expense claim is checked
against two receipt images: the true receipt and a swapped one. Last, bad
inputs are refused before any HTTP call. The script prints each probability
distribution and writes a JSON receipt under ``typevet-receipts/``.

Run with: `uv run python examples/terminal-demo/run.py`

Run the command from the repository root, because the receipt images come from
``tests/fixtures/cord/expense_smoke/``. ``TYPEVET_BACKEND`` selects the backend
(default ``llama_cpp``). For a live llama.cpp run, set
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` to a Gemma 4 vision model id, for example
``gemma-4-31b-kv9-q4km-mm``. ``TYPEVET_BACKEND=fake`` runs offline with uniform
answers.

Examples:
    ```bash
    TYPEVET_BACKEND=fake uv run python examples/terminal-demo/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - [typevet.runtime.open_gemma_native_vision_judgment][]: llama.cpp session.
    - examples/live-demo/README.md: The web page version of this demo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from typevet.adapters.inbound import load_backend, load_llama_settings, open_judgment
from typevet.domain import (
    Choice,
    ImageInput,
    JudgmentResponse,
    JudgmentValidationError,
    Noul,
    Question,
    Score,
    ScoringValidationError,
)
from typevet.runtime import open_gemma_native_vision_judgment

REPO = Path.cwd()
CORD = REPO / "tests" / "fixtures" / "cord" / "expense_smoke"
OUT_DIR = REPO / "typevet-receipts"
WIDTH = 96
BAR = 30
LABEL_W = 34
SMALL_P = 0.0001
HALF = 0.5

CUSTOMER_MESSAGE = (
    "I just saw two charges of $84.20 from the same coffee shop this morning, "
    "I only bought one coffee. Can you fix this?"
)
FRAUD_CRITERIA = {
    "unauthorized_transaction": "A charge the customer did not make or approve",
    "duplicate_charge": "The same purchase was billed more than once",
    "phishing_or_scam": "Customer was tricked into paying or sharing data",
    "account_takeover": "Someone else gained control of the account",
    "not_fraud": "No fraud or billing error is described",
    "unclear": "Not enough information to decide",
}
URGENCY_LEVELS = [
    "none: no money at risk",
    "low: small issue, no money lost",
    "high: money already lost",
    "critical: ongoing loss, act now",
]
VERDICT_CRITERIA = {
    "supported": "The receipt image shows this total",
    "contradicted": "The receipt image shows a different total",
    "insufficient_evidence": "The image does not show enough to decide",
}
TEXT_QUESTIONS: dict[str, Question] = {
    "unauthorized": Noul(
        instructions="Does the customer report a transaction they did not authorize?",
        criteria={
            "true": "Yes, they report a charge they did not authorize",
            "false": "No unauthorized transaction is reported",
        },
    ),
    "fraud_type": Choice(
        instructions="Which type of issue does the customer report?",
        criteria=FRAUD_CRITERIA,
    ),
    "urgency": Score(
        instructions="How urgent is this customer's issue?",
        criteria=URGENCY_LEVELS,
    ),
}
VERDICT_QUESTION = Choice(
    instructions="Look at the receipt image. Does it support the expense claim?",
    criteria=VERDICT_CRITERIA,
)


class Judge(Protocol):
    """The ``judge`` call that each judgment session port gives.

    Examples:
        ```python
        def first_choice(port: Judge, model: str) -> str:
            return port.judge("text", {"q": Choice()}, model).choices["q"].choice
        ```
    """

    def judge(
        self,
        state: str,
        questions: Mapping[str, Question],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Answer each question about ``state``.

        Args:
            state: Content under evaluation.
            questions: Question names to typed questions.
            model: Model id that the session pins.
            media: Images that condition the judgment.
        """
        ...


class CountingTransport(httpx.HTTPTransport):
    """HTTP transport that records each request path in a shared log.

    Attributes:
        log (list[str]): Request lines, one for each request sent.

    Examples:
        ```python
        log: list[str] = []
        transport = CountingTransport(log)
        ```
    """

    def __init__(self, log: list[str]) -> None:
        """Store the shared log.

        Args:
            log: List that receives one line for each request.
        """
        super().__init__()
        self.log = log

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Record the request, then send it.

        Args:
            request: Outgoing request.

        Returns:
            The server response.
        """
        self.log.append(f"{request.method} {request.url.path}")
        return super().handle_request(request)


def rule(ch: str = "=") -> None:
    """Print one horizontal rule.

    Args:
        ch: Character that fills the rule.
    """
    print(ch * WIDTH)


def heading(title: str) -> None:
    """Print a section heading between two rules.

    Args:
        title: Heading text.
    """
    print()
    rule()
    print(f" {title}")
    rule()


def bar(p: float) -> str:
    """Return a text bar for one probability.

    Args:
        p: Probability from 0 to 1.

    Returns:
        A bar of ``BAR`` characters.
    """
    n = round(p * BAR)
    return "#" * n + "." * (BAR - n)


def fmt_p(p: float) -> str:
    """Return one probability as a percentage or in scientific notation.

    Args:
        p: Probability from 0 to 1.

    Returns:
        The formatted probability, eight characters wide.
    """
    if p >= SMALL_P or p == 0.0:
        return f"{p:8.2%}"
    return f"{p:8.1e}"


def print_dist(labels: list[str], probs: Mapping[str, float], winner: str) -> None:
    """Print one distribution with a bar for each label.

    Args:
        labels: Labels in display order.
        probs: Probability for each label.
        winner: Label to mark as the winner.
    """
    for label in labels:
        p = probs[label]
        mark = " <== WINNER" if label == winner else ""
        print(f"    {label:<{LABEL_W}} {fmt_p(p)} |{bar(p)}|{mark}")


def gpu_name() -> str:
    """Return the GPU name and memory from ``nvidia-smi``, or ``unknown``.

    Returns:
        The ``nvidia-smi`` line, or ``unknown`` when the tool is missing.
    """
    try:
        out = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def load_cord_manifest() -> dict[str, Any]:
    """Read the vendored CORD receipt manifest. Repo-only: needs a checkout.

    Returns:
        The parsed ``manifest.json`` of the expense smoke fixtures.
    """
    return json.loads((CORD / "manifest.json").read_text())


@contextmanager
def open_session(
    http_log: list[str], info: dict[str, Any]
) -> Iterator[tuple[Judge, str]]:
    """Open the judgment session that ``TYPEVET_BACKEND`` selects.

    The llama.cpp branch builds its own client so that the demo can count
    router calls and read the build from ``/props``. The other backends use
    ``open_judgment``.

    Args:
        http_log: List that receives one line for each HTTP request.
        info: Mapping that receives the session facts to print and record.

    Yields:
        The session port and its model id.
    """
    backend = load_backend()
    info["backend"] = backend
    if backend != "llama_cpp":
        with open_judgment(transport=CountingTransport(http_log)) as session:
            info["model"] = session.model
            yield session.port, session.model
        return
    settings = load_llama_settings()
    hooks: dict[str, list[Callable[..., Any]]] = {
        "request": [lambda r: http_log.append(f"{r.method} {r.url.path}")]
    }
    with httpx.Client(
        base_url=settings.base_url, timeout=settings.timeout, event_hooks=hooks
    ) as client:
        props = client.get("/props").raise_for_status().json()
        print(
            "  Opening session (router loads the model on first use; may take minutes)..."
        )
        t_open = time.perf_counter()
        with open_gemma_native_vision_judgment(
            settings=settings, http_client=client
        ) as session:
            info.update(
                model=session.model,
                served_template=session.served.value,
                vision_capable=session.capability.vision,
                router=settings.base_url,
                llama_cpp_build=props.get("build_info", "unknown"),
                gpu=gpu_name(),
                session_open_s=round(time.perf_counter() - t_open, 3),
            )
            yield session.port, session.model


def judge_text(port: Judge, model: str, http_log: list[str]) -> dict[str, Any]:
    """Ask the three typed questions about the customer message and print them.

    Args:
        port: Session port.
        model: Model id that the session pins.
        http_log: Shared HTTP request log.

    Returns:
        The receipt entry for the text judgment.
    """
    heading("1. TEXT -- three typed questions on one customer message")
    print("  Customer message:")
    print(f'    "{CUSTOMER_MESSAGE[:73]}')
    print(f'     {CUSTOMER_MESSAGE[73:]}"')
    t0, n0 = time.perf_counter(), len(http_log)
    resp = port.judge(CUSTOMER_MESSAGE, TEXT_QUESTIONS, model)
    text_s, calls = time.perf_counter() - t0, len(http_log) - n0
    a = resp.nouls["unauthorized"]
    print()
    print("  a) Noul (yes/no): Does the customer report a transaction they did")
    print("     not authorize?")
    yes_no = {"yes": a.noul, "no": 1.0 - a.noul}
    print_dist(["yes", "no"], yes_no, "yes" if a.noul >= HALF else "no")
    c = resp.choices["fraud_type"]
    print()
    print("  b) Choice (pick one): Which type of issue does the customer report?")
    print_dist(list(FRAUD_CRITERIA), c.probabilities, c.choice)
    s = resp.scores["urgency"]
    print()
    print("  c) Score (0-3): How urgent is this customer's issue?")
    labels = [f"{i} {level}" for i, level in enumerate(URGENCY_LEVELS)]
    probs = {labels[i]: s.probabilities[i] for i in s.probabilities}
    modal = max(s.probabilities, key=lambda k: s.probabilities[k])
    print_dist(labels, probs, labels[modal])
    print(f"    Expected value (probability-weighted level): {s.score:.2f} / 3")
    print()
    print(
        f"  Text judgment: {text_s:.1f}s, {calls} HTTP calls, "
        f"tokens in/out = {resp.usage.input_tokens}/{resp.usage.output_tokens}"
    )
    return {
        "kind": "text",
        "state": CUSTOMER_MESSAGE,
        "questions": {
            "unauthorized": {
                "type": "noul",
                "instructions": TEXT_QUESTIONS["unauthorized"].instructions,
                "options": ["yes", "no"],
            },
            "fraud_type": {"type": "choice", "options": FRAUD_CRITERIA},
            "urgency": {"type": "score", "levels": URGENCY_LEVELS},
        },
        "answers": {
            "unauthorized": {"p_yes": a.noul},
            "fraud_type": {
                "choice": c.choice,
                "confidence": c.confidence,
                "probabilities": c.probabilities,
            },
            "urgency": {
                "expected_value": s.score,
                "confidence": s.confidence,
                "probabilities": {str(k): v for k, v in s.probabilities.items()},
            },
        },
        "elapsed_s": round(text_s, 3),
        "http_calls": calls,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def judge_images(port: Judge, model: str, http_log: list[str]) -> list[dict[str, Any]]:
    """Check one expense claim against the true and a swapped receipt image.

    Args:
        port: Session port.
        model: Model id that the session pins.
        http_log: Shared HTTP request log.

    Returns:
        One receipt entry for each image case.
    """
    heading("2. IMAGE -- does the receipt image support the expense claim?")
    manifest = load_cord_manifest()
    recs = {r["receipt_id"]: r for r in manifest["receipts"]}
    claims = recs["R01"]["claims"]
    statement = next(c for c in claims if c["expected_verdict"] == "supported")[
        "statement"
    ]
    print(f'  Claim (same for both cases): "{statement}"')
    print(f"  Source: CORD v2 validation receipts ({manifest['license']}), vendored.")
    cases = [
        ("A", "R01", "true receipt for the claim", "supported"),
        ("B", "R02", "SWAPPED: a different receipt", "contradicted"),
    ]
    entries: list[dict[str, Any]] = []
    for case_id, rid, desc, expect in cases:
        rec = recs[rid]
        data = (CORD / rec["image"]["file_name"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        image = ImageInput(data=data, mime_type=rec["image"]["mime_type"])
        t0, n0 = time.perf_counter(), len(http_log)
        r = port.judge(statement, {"verdict": VERDICT_QUESTION}, model, media=(image,))
        el, calls = time.perf_counter() - t0, len(http_log) - n0
        v = r.choices["verdict"]
        print()
        print(f"  Case {case_id}: image {rec['image']['file_name']} ({desc})")
        print(
            f"    receipt's real total (manifest): {rec['annotated_total']}   "
            f"sha256 {sha[:16]}..."
        )
        print(f"    expected answer: {expect}")
        print_dist(list(VERDICT_CRITERIA), v.probabilities, v.choice)
        ok = "MATCHES expectation" if v.choice == expect else "DIFFERS from expectation"
        print(f"    -> model answer: {v.choice} ({ok}); {el:.1f}s, {calls} HTTP calls")
        entries.append(
            {
                "kind": "image",
                "case": case_id,
                "state": statement,
                "question": VERDICT_QUESTION.instructions,
                "options": VERDICT_CRITERIA,
                "image_file": rec["image"]["file_name"],
                "image_sha256": sha,
                "manifest_total": rec["annotated_total"],
                "expected": expect,
                "answer": {
                    "choice": v.choice,
                    "confidence": v.confidence,
                    "probabilities": v.probabilities,
                },
                "elapsed_s": round(el, 3),
                "http_calls": calls,
            }
        )
    return entries


def typed_rejections(
    port: Judge, http_log: list[str], *, pinned: bool
) -> dict[str, Any]:
    """Show that bad inputs are refused before any HTTP call.

    Args:
        port: Session port.
        http_log: Shared HTTP request log.
        pinned: Whether the session pins one model id. The fake does not.

    Returns:
        The receipt entry for the refused inputs.
    """
    heading("3. TYPED GUARANTEE -- bad inputs are rejected before any HTTP call")
    n0 = len(http_log)
    rejects: list[dict[str, str]] = []
    bad_cases: list[tuple[str, Callable[[], object]]] = [
        (
            "ImageInput with mime type image/gif",
            lambda: ImageInput(data=b"GIF89a", mime_type="image/gif"),
        ),
        (
            "ImageInput with empty bytes",
            lambda: ImageInput(data=b"", mime_type="image/png"),
        ),
    ]
    if pinned:
        bad_cases.append(
            (
                "judge() with a model id the session is not pinned to",
                lambda: port.judge(
                    CUSTOMER_MESSAGE, TEXT_QUESTIONS, "some-other-model"
                ),
            )
        )
    for label, fn in bad_cases:
        try:
            fn()
        except (ScoringValidationError, JudgmentValidationError) as exc:
            print(f"  {label}")
            print(f"    -> {type(exc).__name__}:")
            print(f"       {exc}")
            rejects.append(
                {"case": label, "error": type(exc).__name__, "message": str(exc)}
            )
        else:
            print(f"  {label}: NOT rejected (unexpected)")
            rejects.append({"case": label, "error": "none"})
    extra = len(http_log) - n0
    print(f"  HTTP calls made by these {len(bad_cases)} attempts: {extra}")
    return {"cases": rejects, "http_calls": extra}


def main() -> None:
    """Run the three demo sections and write the JSON receipt."""
    t_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    http_log: list[str] = []
    info: dict[str, Any] = {}

    # 1. Open the session that TYPEVET_BACKEND selects and print its facts.
    heading("typevet x Gemma 4 -- typed judgments, live, on local hardware")
    with open_session(http_log, info) as (port, model):
        info["timestamp_utc"] = stamp
        for key, value in info.items():
            print(f"  {key:<17}: {value}")

        # 2. Judge the customer message with three typed questions.
        requests = [judge_text(port, model, http_log)]

        # 3. Judge one expense claim against the true and a swapped receipt image.
        requests += judge_images(port, model, http_log)

        # 4. Show that bad inputs are refused before any HTTP call.
        rejections = typed_rejections(port, http_log, pinned=info["backend"] != "fake")

    # 5. Write the JSON receipt under typevet-receipts/ and print the summary.
    total_s = time.perf_counter() - t_start
    scoring_calls = sum(1 for line in http_log if line.endswith("/completion"))
    receipt = {
        **info,
        "requests": requests,
        "typed_rejections": rejections,
        "total_http_calls": len(http_log),
        "scoring_calls": scoring_calls,
        "http_log": http_log,
        "total_elapsed_s": round(total_s, 3),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"terminal-demo-{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2))
    heading("SUMMARY")
    print(
        f"  HTTP calls to the backend       : {len(http_log)} "
        f"({scoring_calls} scoring /completion, rest probes/tokenize)"
    )
    print(f"  Total elapsed                   : {total_s:.1f}s")
    print("  JSON receipt:")
    print(f"    {out}")
    rule()


if __name__ == "__main__":
    main()
