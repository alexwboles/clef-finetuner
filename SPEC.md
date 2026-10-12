# clef-finetuner — Spec

**Goal:** a free, self-hosted tool that teaches Cloudflare's open-weight decision
models (Clef 27B / Clef-flash 9B, Apache 2.0) your own decision schema, without
Cloudflare's hosted/proprietary RL tuning platform.

## Background (verified Oct 2026)

- Clef (27B, from Qwen3.8-27B) and Clef-flash (9B, from Qwen3.5-9B) are *decision
  models*: they take an input **state** + a schema of **typed questions** and
  return a calibrated probability per allowed answer in one forward pass. No
  text generation, no parsing.
- Question types: `noul` (yes/no), `choice` (one named option), `score`
  (ordered rubric). Up to 64 questions per request; up to 4 images per request.
- Weights on Hugging Face, Apache 2.0. API compatible with TypeSafe AI's
  Jev / System One API. Release helpers: `encode_record`, `collate_records`,
  `load_release_model`, `systemone()` in `joint_schema_model.py`.
- Clef-omni (Oct 9) extends inputs to audio/video — the dataset schema below
  reserves a `media` field for that.
- Clef-flash needs ~41 GB VRAM at full 64k context (Cloudflare PM); with 4-bit
  quantization + LoRA + shorter contexts it trains on a single 24 GB consumer
  GPU. Full-precision 27B Clef is out of scope for the tuner (flash only).

## 1. Dataset format — `clef-dataset.jsonl`

One JSON object per line. Every example:

```json
{
  "id": "guard-001",
  "state": {
    "text": "Run backup of customer DB to s3://backups nightly at 2am",
    "json": {"action": "db.backup", "target": "customer-db", "destructive": false}
  },
  "question": {"type": "noul", "name": "allow"},
  "answer": "yes",
  "media": []
}
```

- `state.text` (required): free-text context. `state.json` (optional): structured
  context. `media` (optional): list of local image/audio/video paths, reserved
  for Clef-omni-style multimodal tuning.
- `question.type`: `noul` | `choice` | `score`.
  - `noul`: `answer` ∈ {"yes","no"}.
  - `choice`: `question.options` = ["a","b","c"], `answer` ∈ options.
  - `score`: `question.levels` = ["low","medium","high"] (ordered),
    `answer` ∈ levels.
- `clef-tuner validate` checks every line: valid JSON, required fields, answer
  ∈ allowed set, no duplicate ids. Reports counts per question type.

## 2. LoRA tuning pipeline — `clef-tuner train`

Input: validated `.jsonl` + `--config train.json` (or CLI flags).

- **Encoding:** each example → Clef's record format via `encode_record`-equivalent:
  state text/JSON serialized + question schema → model inputs; answer → target
  class index. (Mirrors the release's `joint_schema_model.py` helpers.)
- **Real path (GPU machine):** generates a self-contained `run_lora.py`
  (transformers + peft + datasets) + `lora_config.json` (rank 16, alpha 32,
  dropout 0.05, target modules q/k/v/o, 4-bit base via bitsandbytes, bf16,
  3 epochs, lr 2e-4, eval split 10%). User runs it where a GPU exists; the
  adapter is saved to `--out`.
- **Dry-run mode (`--dry-run`):** validates config, prints the exact training
  plan (steps, effective batch, est. VRAM) without touching a model.
- **Mock mode (`--mock`):** exercises the *entire* pipeline on this machine —
  a deterministic pseudo-adapter records per-(question, answer) label
  statistics from the training split, so eval/serving stages run end-to-end
  with no GPU. Mock adapters are clearly labeled and never mistaken for real
  weights.

## 3. Eval rig — `clef-tuner eval`

Input: held-out `.jsonl` + adapter dir (real or mock).

Metrics, per question type and overall:
- **Accuracy** (argmax vs gold answer).
- **Brier score** (mean squared error of predicted probability of the true
  class) — lower is better; decomposes into calibration + refinement.
- **Expected Calibration Error (ECE):** 10-bin |confidence − accuracy|,
  confidence-weighted. The headline calibration number.
- **Confidence histogram** + per-class precision/recall in the JSON report.

Output: `eval_report.json` + a human-readable console table. Exit code
non-zero if accuracy < `--min-accuracy` (default 0.5) so CI can gate.

## 4. Local decision service — `clef-tuner serve`

Drop-in, Jev-API-compatible HTTP service (stdlib only, no dependencies):

```
POST /decide
{"state": {"text": "...", "json": {...}},
 "questions": [{"type": "noul", "name": "allow"},
               {"type": "choice", "name": "route", "options": ["a","b"]}]}

→ {"decisions": [
     {"name": "allow",
      "probabilities": {"yes": 0.93, "no": 0.07},
      "confidence": 0.93, "answer": "yes"},
     {"name": "route",
      "probabilities": {"a": 0.81, "b": 0.19},
      "confidence": 0.81, "answer": "a"}]}
```

- Loads a real LoRA adapter when present, else runs in `--mock` mode with the
  mock adapter — same wire format, so clients can't tell the difference.
- `confidence` = max probability; `answer` = argmax. Calibrated confidence is
  the whole point: downstream code thresholds on it instead of parsing text.
- Health check at `GET /health`.

## 5. Dogfooding — HQ guardrail allow/block

`examples/guardrail_allow_block.jsonl`: 24 labeled examples of HQ agent
actions (tool calls with destructive flags) labeled allow/block by the standing
approval rules. This is the first real schema to tune: the tuned service
becomes a local, auditable second opinion next to the chat-LLM guardrail path,
with agreement rate measured by the eval rig.

## Non-goals

- Full-precision training of 27B Clef (flash only).
- Multimodal training data pipelines (schema reserves `media`; encoders later).
- Cloudflare Workers AI integration (self-hosted only).
- Any paid service, API key, or telemetry. Zero required dependencies for
  validate/eval/serve-mock; torch/transformers/peft only for real GPU runs.
