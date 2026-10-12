# clef-finetuner

Free, self-hosted LoRA fine-tuning + eval rig for Cloudflare's open-weight
**Clef** decision models (Apache 2.0). Teaches Clef-flash your own decision
schema — no hosted platform, no API keys, no cost.

Decision models don't generate text. They take a **state** + typed questions
(`noul` / `choice` / `score`) and return calibrated probabilities per answer.
This tool fine-tunes them on your labeled examples, measures accuracy *and*
calibration, and serves the result as a drop-in Jev-API-compatible endpoint.

## Quickstart (no GPU needed)

```bash
# 1. Validate your dataset
python -m clef_tuner.cli validate examples/guardrail_allow_block.jsonl

# 2. See the training plan (dry run)
python -m clef_tuner.cli train examples/guardrail_allow_block.jsonl \
    --out runs/exp1 --dry-run

# 3. Train a mock adapter (full pipeline, no GPU)
python -m clef_tuner.cli train examples/guardrail_allow_block.jsonl \
    --out runs/exp1 --mock

# 4. Eval on held-out examples
python -m clef_tuner.cli eval examples/guardrail_allow_block.jsonl \
    --model runs/exp1

# 5. Serve it
python -m clef_tuner.cli serve --model runs/exp1 --port 8080
```

Then:
```bash
curl -X POST localhost:8080/decide -H 'Content-Type: application/json' -d '{
  "state": {"text": "Delete ~/workspace/drafts/old.txt"},
  "questions": [{"type": "noul", "name": "allow"}]
}'
```

## Dataset format

JSONL, one example per line — see [SPEC.md](SPEC.md):

```json
{"id": "guard-001",
 "state": {"text": "...", "json": {...}},
 "question": {"type": "noul", "name": "allow"},
 "answer": "yes"}
```

Question types: `noul` (yes/no), `choice` (named options), `score`
(ordered levels). A `media` list reserves multimodal inputs (Clef-omni).

## Real GPU training

`clef-tuner train` (no flags) writes a self-contained `run_lora.py` +
`lora_config.json`. Copy to a GPU box and follow
[docs/GPU_TRAINING.md](docs/GPU_TRAINING.md). Clef-flash fits a 24 GB
consumer GPU at 4-bit.

## Tests

```bash
python -m unittest discover -s tests -v   # smoke + e2e
```

## License

MIT. Clef model weights are Apache 2.0 (Cloudflare); this tool ships no
weights and is independent of Cloudflare.
