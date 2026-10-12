# Running real LoRA training on a GPU machine

`clef-tuner train --mock` exercises the whole pipeline with no GPU. This doc
covers the real thing: LoRA-tuning Clef-flash (9B) on your own examples.

## What you need

- A machine with a CUDA GPU. 24 GB VRAM is enough at 4-bit quantization
  (the default config); 41 GB+ if you want longer contexts or higher rank.
- Python 3.10+, ~30 GB disk for the base weights.

## Steps

1. On this machine, generate the training bundle:
   ```
   clef-tuner train data.jsonl --out runs/exp1
   ```
   This writes `runs/exp1/lora_config.json` and `runs/exp1/run_lora.py`.

2. Copy `runs/exp1/` and `data.jsonl` to the GPU machine.

3. On the GPU machine:
   ```
   pip install torch transformers peft bitsandbytes datasets trl
   python runs/exp1/run_lora.py --data data.jsonl --out runs/exp1/adapter
   ```

4. Copy `runs/exp1/adapter/` back. Point the eval rig and server at it:
   ```
   clef-tuner eval data.jsonl --model runs/exp1/adapter
   clef-tuner serve --model runs/exp1/adapter --port 8080
   ```

## Notes

- Base weights: `cloudflare/clef-flash` on Hugging Face (Apache 2.0).
  The generated script mirrors the release's `joint_schema_model.py` record
  encoding; if your checkpoint ships updated helpers, adapt `encode()`.
- Start with the defaults (r=16, 3 epochs, lr 2e-4). If eval ECE is high,
  train longer before touching the rank.
- The mock adapter (`--mock`) and real adapters share the same eval and
  serve code paths — a pipeline that passes `--mock` end-to-end will work
  unchanged with real weights.
- Multimodal (`media` field): encoding for image/audio inputs follows the
  Clef-omni release helpers; text-only tuning works today, media encoders
  are on the roadmap.
