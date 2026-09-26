from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
HOST = os.getenv("RERANKER_HOST", "0.0.0.0")
PORT = int(os.getenv("RERANKER_PORT", "8080"))
BATCH_SIZE = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "8")))
MAX_LENGTH = max(64, int(os.getenv("RERANKER_MAX_LENGTH", "512")))
DEVICE = os.getenv("RERANKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOCK = Lock()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


def score_pairs(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    with MODEL_LOCK, torch.inference_mode():
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start : start + BATCH_SIZE]
            inputs = tokenizer(
                [query] * len(batch),
                [str(item.get("text", "")) for item in batch],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            logits = model(**inputs).logits.reshape(-1).detach().float().cpu().tolist()
            scored.extend(
                {"id": str(item["id"]), "score": float(score), "_index": start + offset}
                for offset, (item, score) in enumerate(zip(batch, logits))
            )
    scored.sort(key=lambda item: (-item["score"], item["_index"]))
    return [{"id": item["id"], "score": item["score"]} for item in scored]


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": MODEL_NAME})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            query = str(payload["query"])
            candidates = payload["candidates"]
            if not isinstance(candidates, list) or any("id" not in item for item in candidates):
                raise ValueError("candidates must be a list containing ids")
            self._send(200, {"results": score_pairs(query, candidates)})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - protects the service boundary
            self._send(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
