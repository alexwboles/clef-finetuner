"""Dataset loading and validation for clef-dataset.jsonl.

One JSON object per line:
  {"id": ..., "state": {"text": ..., "json": {...}, "media": [...]},
   "question": {"type": "noul"|"choice"|"score", "name": ..., ...},
   "answer": ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass
class Example:
    id: str
    state_text: str
    state_json: dict
    media: list
    qtype: str
    qname: str
    options: list  # allowed answers in order
    answer: str
    raw: dict = field(repr=False)

    @property
    def answer_index(self) -> int:
        return self.options.index(self.answer)


def allowed_answers(question: dict) -> list[str]:
    qtype = question.get("type")
    if qtype == "noul":
        return ["yes", "no"]
    if qtype == "choice":
        return list(question.get("options") or [])
    if qtype == "score":
        return list(question.get("levels") or [])
    return []


def validate_example(obj: Any, lineno: int) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return [f"line {lineno}: not a JSON object"]
    if not obj.get("id"):
        errs.append(f"line {lineno}: missing 'id'")
    state = obj.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("text"), str):
        errs.append(f"line {lineno}: state.text must be a string")
    media = obj.get("media", [])
    if media is None:
        media = []
    if not isinstance(media, list):
        errs.append(f"line {lineno}: media must be a list")
    q = obj.get("question")
    if not isinstance(q, dict):
        errs.append(f"line {lineno}: missing question object")
        return errs
    if q.get("type") not in QUESTION_TYPES:
        errs.append(f"line {lineno}: question.type must be one of {QUESTION_TYPES}")
    if not isinstance(q.get("name"), str) or not q.get("name"):
        errs.append(f"line {lineno}: question.name must be a non-empty string")
    allowed = allowed_answers(q)
    if q.get("type") in ("choice", "score") and not allowed:
        errs.append(f"line {lineno}: {q.get('type')} needs non-empty options/levels")
    answer = obj.get("answer")
    if answer not in allowed:
        errs.append(f"line {lineno}: answer {answer!r} not in {allowed}")
    return errs


def load_dataset(path: str) -> tuple[list[Example], list[str]]:
    """Load a .jsonl dataset. Returns (examples, errors)."""
    examples: list[Example] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue
            errs = validate_example(obj, lineno)
            ex_id = obj.get("id") if isinstance(obj, dict) else None
            if ex_id in seen_ids:
                errs.append(f"line {lineno}: duplicate id {ex_id!r}")
            errors.extend(errs)
            if errs:
                continue
            seen_ids.add(ex_id)
            q = obj["question"]
            examples.append(
                Example(
                    id=ex_id,
                    state_text=obj["state"]["text"],
                    state_json=obj["state"].get("json") or {},
                    media=obj.get("media") or [],
                    qtype=q["type"],
                    qname=q["name"],
                    options=allowed_answers(q),
                    answer=obj["answer"],
                    raw=obj,
                )
            )
    return examples, errors


def split_dataset(
    examples: list[Example], eval_frac: float = 0.1, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Deterministic train/eval split (eval_frac of examples, at least 1)."""
    import random

    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_eval = max(1, int(round(len(examples) * eval_frac))) if examples else 0
    eval_idx = set(idx[:n_eval])
    train = [e for i, e in enumerate(examples) if i not in eval_idx]
    ev = [e for i, e in enumerate(examples) if i in eval_idx]
    return train, ev
