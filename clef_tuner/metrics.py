"""Eval metrics for decision models: accuracy, Brier score, ECE.

All functions take plain Python lists so the eval rig stays dependency-free.
"""
from __future__ import annotations


def accuracy(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def brier_score(gold: list[int], probas: list[list[float]]) -> float:
    """Mean squared error between predicted probas and one-hot gold."""
    if not gold:
        return 0.0
    total = 0.0
    for g, pr in zip(gold, probas):
        for k, p in enumerate(pr):
            total += (p - (1.0 if k == g else 0.0)) ** 2
    return total / len(gold)


def expected_calibration_error(
    gold: list[int], probas: list[list[float]], n_bins: int = 10
) -> float:
    """ECE: |confidence - accuracy| per confidence bin, weighted by bin size."""
    if not gold:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for g, pr in zip(gold, probas):
        conf = max(pr)
        correct = pr.index(conf) == g
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    ece = 0.0
    n = len(gold)
    for b in bins:
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        mean_acc = sum(1.0 for _, ok in b if ok) / len(b)
        ece += (len(b) / n) * abs(mean_conf - mean_acc)
    return ece


def confidence_histogram(
    probas: list[list[float]], n_bins: int = 10
) -> list[int]:
    hist = [0] * n_bins
    for pr in probas:
        b = min(int(max(pr) * n_bins), n_bins - 1)
        hist[b] += 1
    return hist


def per_class_prf(
    gold: list[int], pred: list[int], n_classes: int
) -> list[dict]:
    out = []
    for c in range(n_classes):
        tp = sum(1 for g, p in zip(gold, pred) if p == c and g == c)
        fp = sum(1 for g, p in zip(gold, pred) if p == c and g != c)
        fn = sum(1 for g, p in zip(gold, pred) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append(
            {"class": c, "precision": prec, "recall": rec, "f1": f1,
             "support": sum(1 for g in gold if g == c)}
        )
    return out
