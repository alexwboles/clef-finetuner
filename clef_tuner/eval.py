"""Eval rig: accuracy + calibration metrics on held-out examples."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from .dataset import Example, split_dataset
from .metrics import (
    accuracy,
    brier_score,
    confidence_histogram,
    expected_calibration_error,
    per_class_prf,
)
from .mock import MockAdapter


def run_eval(
    examples: list[Example],
    adapter_dir: str,
    eval_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    _, ev = split_dataset(examples, eval_frac=eval_frac, seed=seed)
    adapter = MockAdapter.load(adapter_dir)
    gold, pred, probas = [], [], []
    per_type: dict[str, dict[str, list]] = defaultdict(
        lambda: {"gold": [], "pred": [], "probas": []}
    )
    for ex in ev:
        pr = adapter.predict_proba(ex)
        p = pr.index(max(pr))
        gold.append(ex.answer_index)
        pred.append(p)
        probas.append(pr)
        bucket = per_type[ex.qtype]
        bucket["gold"].append(ex.answer_index)
        bucket["pred"].append(p)
        bucket["probas"].append(pr)

    def summarize(g, p, pr):
        n_classes = max(len(x) for x in pr) if pr else 0
        return {
            "n": len(g),
            "accuracy": round(accuracy(g, p), 4),
            "brier": round(brier_score(g, pr), 4),
            "ece": round(expected_calibration_error(g, pr), 4),
            "conf_hist": confidence_histogram(pr),
            "per_class": per_class_prf(g, p, n_classes),
        }

    report = {
        "n_eval": len(ev),
        "overall": summarize(gold, pred, probas),
        "by_type": {
            t: summarize(b["gold"], b["pred"], b["probas"])
            for t, b in per_type.items()
        },
    }
    return report


def save_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def format_report(report: dict) -> str:
    o = report["overall"]
    lines = [
        f"eval on {report['n_eval']} held-out examples",
        f"  accuracy : {o['accuracy']:.3f}",
        f"  brier    : {o['brier']:.3f}  (lower is better)",
        f"  ECE      : {o['ece']:.3f}  (lower is better)",
        "  by type:",
    ]
    for t, s in report["by_type"].items():
        lines.append(
            f"    {t:<7} n={s['n']:<4} acc={s['accuracy']:.3f} "
            f"brier={s['brier']:.3f} ece={s['ece']:.3f}"
        )
    return "\n".join(lines)
