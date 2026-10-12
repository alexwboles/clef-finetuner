"""E2E tests: full pipeline validate -> train(mock) -> eval -> serve.

Uses the real CLI entry points and a live HTTP server on localhost.
No GPU, no network beyond loopback.
"""
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clef_tuner.cli import main as cli_main
from clef_tuner.server import start_background

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "guardrail_allow_block.jsonl")


def run_cli(*argv):
    return cli_main(list(argv))


class TestPipeline(unittest.TestCase):
    def test_validate_example_dataset(self):
        self.assertEqual(run_cli("validate", EXAMPLES), 0)

    def test_dry_run_prints_plan(self):
        with tempfile.TemporaryDirectory() as d:
            rc = run_cli("train", EXAMPLES, "--out", d, "--dry-run")
            self.assertEqual(rc, 0)
            # dry-run writes nothing
            self.assertEqual(os.listdir(d), [])

    def test_mock_train_writes_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            rc = run_cli("train", EXAMPLES, "--out", d, "--mock")
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "mock_adapter.json")))

    def test_real_bundle_generation(self):
        with tempfile.TemporaryDirectory() as d:
            rc = run_cli("train", EXAMPLES, "--out", d)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "lora_config.json")))
            self.assertTrue(os.path.exists(os.path.join(d, "run_lora.py")))
            cfg = json.load(open(os.path.join(d, "lora_config.json")))
            self.assertEqual(cfg["lora_r"], 16)

    def test_eval_produces_report(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("train", EXAMPLES, "--out", d, "--mock"), 0)
            rc = run_cli("eval", EXAMPLES, "--model", d, "--min-accuracy", "0.0")
            self.assertEqual(rc, 0)
            report_path = os.path.join(d, "eval_report.json")
            self.assertTrue(os.path.exists(report_path))
            report = json.load(open(report_path))
            self.assertIn("overall", report)
            self.assertIn("by_type", report)
            self.assertGreaterEqual(report["overall"]["accuracy"], 0.0)
            self.assertLessEqual(report["overall"]["ece"], 1.0)

    def test_eval_min_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("train", EXAMPLES, "--out", d, "--mock"), 0)
            # impossibly high bar -> exit code 2
            rc = run_cli("eval", EXAMPLES, "--model", d, "--min-accuracy", "1.01")
            self.assertEqual(rc, 2)

    def test_invalid_dataset_fails_validate(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        fh.write('{"id": "bad", "question": {"type": "noul", "name": "x"}, '
                 '"answer": "maybe"}\n')
        fh.close()
        try:
            self.assertEqual(run_cli("validate", fh.name), 1)
        finally:
            os.unlink(fh.name)


class TestServe(unittest.TestCase):
    def _post(self, port, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/decide",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    def test_decide_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("train", EXAMPLES, "--out", d, "--mock"), 0)
            server, _ = start_background(d, port=0)
            port = server.server_address[1]
            try:
                status, body = self._post(port, {
                    "state": {"text": "Read ~/workspace/GOAL.md"},
                    "questions": [
                        {"type": "noul", "name": "allow"},
                        {"type": "choice", "name": "route",
                         "options": ["a", "b"]},
                    ],
                })
                self.assertEqual(status, 200)
                self.assertEqual(len(body["decisions"]), 2)
                for dec in body["decisions"]:
                    self.assertAlmostEqual(
                        sum(dec["probabilities"].values()), 1.0, places=3)
                    self.assertIn(dec["answer"], dec["probabilities"])
                    self.assertGreaterEqual(dec["confidence"], 0.0)
                    self.assertLessEqual(dec["confidence"], 1.0)
            finally:
                server.shutdown()

    def test_health(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("train", EXAMPLES, "--out", d, "--mock"), 0)
            server, _ = start_background(d, port=0)
            port = server.server_address[1]
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=10) as resp:
                    body = json.loads(resp.read())
                self.assertEqual(body["status"], "ok")
            finally:
                server.shutdown()

    def test_bad_question_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run_cli("train", EXAMPLES, "--out", d, "--mock"), 0)
            server, _ = start_background(d, port=0)
            port = server.server_address[1]
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/decide",
                    data=json.dumps({"state": {}, "questions": [
                        {"type": "bogus", "name": "x"}]}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req, timeout=10)
                self.assertEqual(ctx.exception.code, 400)
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
