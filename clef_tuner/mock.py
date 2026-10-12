"""Deterministic mock backend.

Lets the full pipeline (train -> eval -> serve) run end-to-end with no GPU.
A mock adapter records per-(question-name, answer) label statistics from the
training split; predictions are a smoothed version of those statistics,
perturbed deterministically by a hash of the state text so individual
examples vary but runs are reproducible.

Mock adapters are JSON files clearly labeled {"kind": "mock"} and are never
mistaken for real LoRA weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .dataset import Example

MOCK_KIND = "mock"
ADAPTER_FILE = "mock_adapter.json"


def _hash01(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockAdapter:
    def __init__(self, stats: dict[str, Any]):
        self.stats = stats  # {qname: {"options": [...], "counts": [...]}} 

    @classmethod
    def train(cls, examples: list[Example]) -> "MockAdapter":
        stats: dict[str, Any] = {}
        for ex in examples:
            s = stats.setdefault(
                ex.qname, {"options": ex.options, "counts": [0] * len(ex.options)}
            )
            # If the same question name appears with different options, keep
            # the first-seen option list (datasets are schema-consistent).
            s["counts"][ex.answer_index] += 1
        return cls(stats)

    def predict_proba(self, example: Example, smoothing: float = 0.15) -> list[float]:
        s = self.stats.get(example.qname)
        options = example.options
        if s is None or s["options"] != options:
            # Unseen question: uniform prior, perturbed by state hash.
            base = [1.0 / len(options)] * len(options)
        else:
            total = sum(s["counts"]) or 1
            base = [
                (c + smoothing) / (total + smoothing * len(options))
                for c in s["counts"]
            ]
        # Deterministic per-example perturbation so the eval isn't degenerate.
        perturbed = [
            p * (0.9 + 0.2 * _hash01(example.id, example.qname, str(i)))
            for i, p in enumerate(base)
        ]
        z = sum(perturbed)
        return [p / z for p in perturbed]

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        payload = {"kind": MOCK_KIND, "stats": self.stats}
        path = os.path.join(out_dir, ADAPTER_FILE)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    @classmethod
    def load(cls, in_dir: str) -> "MockAdapter":
        path = os.path.join(in_dir, ADAPTER_FILE)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("kind") != MOCK_KIND:
            raise ValueError(f"{path} is not a mock adapter")
        return cls(payload["stats"])
