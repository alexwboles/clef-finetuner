"""Smoke tests: unit-level checks for every module. No GPU, no network."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clef_tuner import dataset, metrics
from clef_tuner.dataset import allowed_answers, load_dataset, split_dataset, validate_example
from clef_tuner.mock import MockAdapter
from clef_tuner.train import DEFAULTS, build_plan


def write_jsonl(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


def ex(i, qtype="noul", answer="yes"):
    q = {"type": "noul", "name": "allow"}
    if qtype == "choice":
        q = {"type": "choice", "name": "route", "options": ["a", "b"]}
    if qtype == "score":
        q = {"type": "score", "name": "risk", "levels": ["low", "high"]}
    return {"id": f"e{i}", "state": {"text": f"state {i}"},
            "question": q, "answer": answer}


class TestDataset(unittest.TestCase):
    def test_valid_noul(self):
        self.assertEqual(validate_example(ex(1), 1), [])

    def test_valid_choice_and_score(self):
        self.assertEqual(validate_example(ex(2, "choice", "a"), 2), [])
        self.assertEqual(validate_example(ex(3, "score", "high"), 3), [])

    def test_bad_answer_rejected(self):
        errs = validate_example(ex(4, "choice", "zzz"), 4)
        self.assertTrue(any("not in" in e for e in errs))

    def test_missing_fields_rejected(self):
        errs = validate_example({"id": "x"}, 9)
        self.assertGreater(len(errs), 0)

    def test_bad_json_line_reported(self):
        path = write_jsonl([ex(1)])
        with open(path, "a") as fh:
            fh.write("{not json\n")
        _, errors = load_dataset(path)
        self.assertTrue(any("invalid JSON" in e for e in errors))
        os.unlink(path)

    def test_duplicate_id_rejected(self):
        path = write_jsonl([ex(1), ex(1)])
        _, errors = load_dataset(path)
        self.assertTrue(any("duplicate id" in e for e in errors))
        os.unlink(path)

    def test_allowed_answers(self):
        self.assertEqual(allowed_answers({"type": "noul"}), ["yes", "no"])
        self.assertEqual(allowed_answers({"type": "choice", "options": ["a"]}), ["a"])
        self.assertEqual(allowed_answers({"type": "score", "levels": ["l", "h"]}), ["l", "h"])

    def test_split_deterministic(self):
        rows = [ex(i) for i in range(20)]
        path = write_jsonl(rows)
        examples, errors = load_dataset(path)
        self.assertEqual(errors, [])
        t1, e1 = split_dataset(examples, eval_frac=0.2, seed=7)
        t2, e2 = split_dataset(examples, eval_frac=0.2, seed=7)
        self.assertEqual([e.id for e in e1], [e.id for e in e2])
        self.assertEqual(len(e1), 4)
        self.assertEqual(len(t1) + len(e1), 20)
        os.unlink(path)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(metrics.accuracy([0, 1, 1], [0, 1, 0]), 2 / 3)

    def test_brier_perfect(self):
        self.assertAlmostEqual(
            metrics.brier_score([0, 1], [[1.0, 0.0], [0.0, 1.0]]), 0.0)

    def test_brier_worst(self):
        self.assertAlmostEqual(
            metrics.brier_score([0], [[0.0, 1.0]]), 2.0)

    def test_ece_perfectly_calibrated(self):
        # All predictions 1.0 and correct -> ECE 0.
        self.assertAlmostEqual(
            metrics.expected_calibration_error([0, 0], [[1.0, 0.0], [1.0, 0.0]]),
            0.0)

    def test_ece_overconfident(self):
        # Always predicts 1.0, always wrong -> ECE 1.0.
        self.assertAlmostEqual(
            metrics.expected_calibration_error([1, 1], [[1.0, 0.0], [1.0, 0.0]]),
            1.0)

    def test_confidence_histogram_sums(self):
        hist = metrics.confidence_histogram([[0.9, 0.1], [0.5, 0.5]])
        self.assertEqual(sum(hist), 2)

    def test_per_class_prf(self):
        prf = metrics.per_class_prf([0, 0, 1], [0, 1, 1], 2)
        self.assertEqual(prf[0]["support"], 2)
        self.assertEqual(prf[1]["support"], 1)


class TestMockAdapter(unittest.TestCase):
    def _examples(self):
        rows = [ex(i, "choice", "a" if i % 2 == 0 else "b") for i in range(10)]
        path = write_jsonl(rows)
        examples, _ = load_dataset(path)
        os.unlink(path)
        return examples

    def test_train_predict_roundtrip(self):
        examples = self._examples()
        adapter = MockAdapter.train(examples)
        with tempfile.TemporaryDirectory() as d:
            saved = adapter.save(d)
            self.assertTrue(os.path.exists(saved))
            loaded = MockAdapter.load(d)
        proba = loaded.predict_proba(examples[0])
        self.assertAlmostEqual(sum(proba), 1.0)
        self.assertEqual(len(proba), 2)

    def test_deterministic(self):
        examples = self._examples()
        adapter = MockAdapter.train(examples)
        p1 = adapter.predict_proba(examples[3])
        p2 = adapter.predict_proba(examples[3])
        self.assertEqual(p1, p2)

    def test_unseen_question_uniform(self):
        examples = self._examples()
        adapter = MockAdapter.train(examples)
        other = ex(99, "noul", "yes")
        path = write_jsonl([other])
        (other_ex,), _ = load_dataset(path)
        os.unlink(path)
        proba = adapter.predict_proba(other_ex)
        self.assertAlmostEqual(sum(proba), 1.0)
        self.assertEqual(len(proba), 2)

    def test_wrong_kind_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "mock_adapter.json"), "w") as fh:
                json.dump({"kind": "real"}, fh)
            with self.assertRaises(ValueError):
                MockAdapter.load(d)


class TestTrainPlan(unittest.TestCase):
    def test_plan_math(self):
        rows = [ex(i) for i in range(20)]
        path = write_jsonl(rows)
        examples, _ = load_dataset(path)
        os.unlink(path)
        cfg = dict(DEFAULTS)
        plan = build_plan(examples, cfg)
        self.assertEqual(plan.n_train + plan.n_eval, 20)
        self.assertGreater(plan.total_steps, 0)
        desc = plan.describe()
        self.assertIn("Clef-flash", desc)
        self.assertIn("LoRA", desc)


if __name__ == "__main__":
    unittest.main()
