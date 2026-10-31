from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from serviceclients.completion_evaluator.models import CompletionEvaluatorResult

DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MODEL_NAME = os.getenv("COMPLETION_EVALUATOR_MODEL", DEFAULT_MODEL_NAME)
HOST = os.getenv("COMPLETION_EVALUATOR_HOST", "0.0.0.0")
PORT = int(os.getenv("COMPLETION_EVALUATOR_PORT", "8082"))
MAX_LENGTH = max(64, int(os.getenv("COMPLETION_EVALUATOR_MAX_LENGTH", "512")))
DEVICE = os.getenv("COMPLETION_EVALUATOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOCK = Lock()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


def _label_name(label: str) -> str:
    normalized = label.lower().replace("_", " ").replace("-", " ")
    if "entail" in normalized:
        return "entailment"
    if "contrad" in normalized:
        return "contradiction"
    return "neutral"


def evaluate_pair(premise: str, hypothesis: str) -> CompletionEvaluatorResult:
    with MODEL_LOCK, torch.inference_mode():
        inputs = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1).cpu().tolist()

    scores = {"contradiction": 0.0, "neutral": 0.0, "entailment": 0.0}
    for index, probability in enumerate(probabilities):
        label = _label_name(str(model.config.id2label.get(index, index)))
        scores[label] += float(probability)
    return CompletionEvaluatorResult(**scores)


class Handler(BaseHTTPRequestHandler):
    def _send_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_response(200, {"status": "ok", "model": MODEL_NAME})
            return
        self._send_response(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/evaluate":
            self._send_response(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            premise = str(payload["premise"])
            hypothesis = str(payload["hypothesis"])
            self._send_response(200, evaluate_pair(premise, hypothesis).model_dump(mode="json"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_response(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - protects the service boundary
            self._send_response(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
