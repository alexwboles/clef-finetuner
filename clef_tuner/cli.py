#!/usr/bin/env python3
"""clef-tuner CLI: validate, train, eval, serve decision-model adapters.

Free, self-hosted LoRA fine-tuning + eval rig for Cloudflare Clef-flash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .dataset import load_dataset
from .eval import format_report, run_eval, save_report
from .train import DEFAULTS, build_plan, generate_real_training, train_mock


def cmd_validate(args: argparse.Namespace) -> int:
    examples, errors = load_dataset(args.data)
    if errors:
        print(f"{len(errors)} error(s) in {args.data}:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1
    by_type: dict[str, int] = {}
    for ex in examples:
        by_type[ex.qtype] = by_type.get(ex.qtype, 0) + 1
    print(f"valid: {len(examples)} examples "
          f"({', '.join(f'{t}={n}' for t, n in sorted(by_type.items()))})")
    return 0


def _config_from_args(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    for key in ("lora_r", "epochs", "batch_size", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if getattr(args, "lr", None) is not None:
        cfg["lr"] = args.lr
    return cfg


def cmd_train(args: argparse.Namespace) -> int:
    examples, errors = load_dataset(args.data)
    if errors:
        print(f"dataset has {len(errors)} error(s); run 'validate' first")
        return 1
    if not examples:
        print("dataset is empty")
        return 1
    cfg = _config_from_args(args)
    plan = build_plan(examples, cfg)
    if args.dry_run:
        print(plan.describe())
        print("\ndry-run: no model touched, no files written.")
        return 0
    os.makedirs(args.out, exist_ok=True)
    if args.mock:
        path = train_mock(examples, args.out, cfg)
        print(plan.describe())
        print(f"\nmock adapter written to {path}")
        print("NOTE: mock adapter is for pipeline testing only, not real weights.")
        return 0
    files = generate_real_training(args.out, cfg)
    print(plan.describe())
    print("\nGPU training bundle written:")
    for f in files:
        print(f"  {f}")
    print("\nCopy the bundle + dataset to a GPU machine and run:")
    print("  pip install torch transformers peft bitsandbytes datasets trl")
    print(f"  python {os.path.join(args.out, 'run_lora.py')} "
          f"--data {args.data} --out {args.out}/adapter")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    examples, errors = load_dataset(args.data)
    if errors:
        print(f"dataset has {len(errors)} error(s); run 'validate' first")
        return 1
    cfg = _config_from_args(args)
    report = run_eval(examples, args.model, eval_frac=cfg["eval_frac"],
                      seed=cfg["seed"])
    path = save_report(report, args.model)
    print(format_report(report))
    print(f"\nreport saved to {path}")
    if report["overall"]["accuracy"] < args.min_accuracy:
        print(f"accuracy below --min-accuracy {args.min_accuracy}")
        return 2
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve_forever

    serve_forever(args.model, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="clef-tuner",
        description="Free, self-hosted fine-tuning + eval for Clef decision models.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="validate a .jsonl dataset")
    p.add_argument("data")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("train", help="train a LoRA adapter (or plan/mock it)")
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="eval an adapter on held-out examples")
    p.add_argument("data")
    p.add_argument("--model", required=True, help="adapter dir from train")
    p.add_argument("--config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-accuracy", type=float, default=0.5)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="serve an adapter as a Jev-compatible API")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
