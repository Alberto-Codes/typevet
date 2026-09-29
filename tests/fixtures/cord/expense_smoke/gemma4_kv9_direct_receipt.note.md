# Identity caveats for `gemma4_kv9_direct_receipt.json`

Kind: reference (fixture annotation).

This note documents known lies in `experiment_identity` for the vendored Gemma 4
direct-arm receipt. Measured semantic labels and FAIL statuses in the JSON are
preserved as recorded; only identity fields are unreliable.

## Caveats

1. **`working_tree.dirty` was falsely `false`.** Identity used a HEAD-only or
   incomplete working-tree capture, so local edits at run time were not
   reflected.
2. **Identity was captured after scoring finished**, not before the first
   request. Digests and `run_id` therefore do not bound the code that actually
   scored the arms.
3. **`code_path_digests` omits the live smoke harness path** (for example
   `tests/live/test_cord_expense_smoke_live.py`, now under `evals/`). Receipt digests do not prove
   which test module drove the run.
4. **`runtime.server_build` is `unknown`.** Router build identity was not
   recorded on this historical run.

For offline regression, treat capability fields at the receipt root
(`vision`, `served_template`, `model`) and attachment token counts as the
smoke contract. Treat `experiment_identity` as archival metadata with the
caveats above.

See also: [Run the CORD expense smoke](../../../../docs/how-to/run-the-cord-expense-smoke.md)
(Historical run, Gemma 4).
