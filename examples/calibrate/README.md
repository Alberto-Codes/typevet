# Calibrate

Kind: reference. This page gives the command and the output shape of the calibrate example.

The script reads the Platt calibration map in `tests/fixtures/calibration/platt_noul_map.json`. `CalibratedJudgment` wraps the session port with that map. One Noul asks whether a short message is a fraud attempt. The map changes the raw probability into a calibrated probability.

Run from the repository root:

```console
uv run python examples/calibrate/run.py
```

`TYPEVET_BACKEND` selects the backend. The default is `llama_cpp`. The wrapper refuses a map whose task, backend or model differs from the call. This example reads the task, backend and model from the fixture map, so the checks pass by construction. In your own code pass your task id, `load_backend()` and `session.model`. The fixture map is synthetic and names the fake model, so run it with `TYPEVET_BACKEND=fake`. That run is offline, with a uniform raw answer. For a live run, use a map fitted on your model and backend. The script writes no file.

The output has the message, the raw and the calibrated probability, and the winner on the last line. The winner is `yes` when the calibrated probability is more than 0.5, and `no` in other cases:

```text
message: Your account is locked. Reply with your card PIN to unlock it today.
raw: <probability>
calibrated: <probability>
winner: <yes or no>
```
