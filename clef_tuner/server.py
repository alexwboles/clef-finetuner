"""Local decision service — Jev-API-compatible, stdlib only.

POST /decide  {"state": {"text":..., "json": {...}},
               "questions": [{"type":"noul","name":"allow"},
                             {"type":"choice","name":"route","options":[...]}]}
→ {"decisions": [{"name":..., "probabilities": {...},
                  "confidence": ..., "answer": ...}]}
GET /health   → {"status": "ok", "model": ..., "mode": "mock"|"real"}
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from .dataset import Example, allowed_answers
from .mock import MockAdapter


def _coerce_question(q: dict) -> tuple[str, str, list[str]]:
    qtype = q.get("type")
    name = q.get("name", "")
    if qtype not in ("noul", "choice", "score"):
        raise ValueError(f"unsupported question type {qtype!r}")
    options = allowed_answers(q)
    if not options:
        raise ValueError(f"question {name!r} has no allowed answers")
    if not name:
        raise ValueError("question needs a name")
    return qtype, name, options


class DecisionService:
    def __init__(self, adapter_dir: str, model_name: str = "clef-flash-tuned"):
        self.adapter = MockAdapter.load(adapter_dir)
        self.model_name = model_name
        self.mode = "mock"  # real-adapter loading plugs in here

    def decide(self, state: dict, questions: list[dict]) -> list[dict]:
        text = state.get("text", "") if isinstance(state, dict) else ""
        out = []
        for q in questions:
            qtype, name, options = _coerce_question(q)
            ex = Example(
                id="req", state_text=text, state_json=state.get("json", {}),
                media=[], qtype=qtype, qname=name, options=options,
                answer=options[0], raw={},
            )
            probas = self.adapter.predict_proba(ex)
            best = probas.index(max(probas))
            out.append({
                "name": name,
                "probabilities": {o: round(p, 4) for o, p in zip(options, probas)},
                "confidence": round(max(probas), 4),
                "answer": options[best],
            })
        return out


def make_handler(service: DecisionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-tuner/0.1.0"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/health":
                self._send(200, {"status": "ok", "model": service.model_name,
                                 "mode": service.mode})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/decide":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON"})
                return
            state = payload.get("state", {})
            questions = payload.get("questions", [])
            if not isinstance(questions, list) or not questions:
                self._send(400, {"error": "questions must be a non-empty list"})
                return
            try:
                decisions = service.decide(state, questions)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"decisions": decisions})

        def log_message(self, *args):  # keep serve output clean
            pass

    return Handler


def run_server(adapter_dir: str, host: str = "127.0.0.1", port: int = 8080,
               model_name: str = "clef-flash-tuned") -> HTTPServer:
    service = DecisionService(adapter_dir, model_name)
    server = HTTPServer((host, port), make_handler(service))
    return server


def serve_forever(adapter_dir: str, host: str = "127.0.0.1", port: int = 8080,
                  model_name: str = "clef-flash-tuned") -> None:
    server = run_server(adapter_dir, host, port, model_name)
    print(f"clef-tuner serving {model_name} (mock) on http://{host}:{port}")
    print("POST /decide  |  GET /health  — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def start_background(adapter_dir: str, host: str = "127.0.0.1", port: int = 0,
                     model_name: str = "test") -> tuple[HTTPServer, threading.Thread]:
    """Start the server on a background thread (used by e2e tests)."""
    server = run_server(adapter_dir, host, port, model_name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
