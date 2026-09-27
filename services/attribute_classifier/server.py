from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from setfit import SetFitModel
from services.attribute_classifier.models import (
    AttributeCandidate,
    AttributeClassificationRequest,
    AttributeClassificationResponse,
    AttributeClassificationResult,
    AttributeValue,
)

logging.basicConfig(level=os.getenv("ATTRIBUTE_CLASSIFIER_LOG_LEVEL", "INFO"))
logger = logging.getLogger("attribute_classifier.server")

DEFAULT_MODEL_NAME = "your-org/attribute-relevance-setfit"
MODEL_NAME = os.getenv("ATTRIBUTE_CLASSIFIER_MODEL", DEFAULT_MODEL_NAME)
HOST = os.getenv("ATTRIBUTE_CLASSIFIER_HOST", "0.0.0.0")
PORT = int(os.getenv("ATTRIBUTE_CLASSIFIER_PORT", "8081"))
BATCH_SIZE = max(1, int(os.getenv("ATTRIBUTE_CLASSIFIER_BATCH_SIZE", "32")))
THRESHOLD = float(os.getenv("ATTRIBUTE_CLASSIFIER_THRESHOLD", "0.5"))
DEVICE = os.getenv("ATTRIBUTE_CLASSIFIER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOCK = Lock()


def _load_model() -> SetFitModel:
    if MODEL_NAME == DEFAULT_MODEL_NAME:
        raise RuntimeError(
            "ATTRIBUTE_CLASSIFIER_MODEL is not configured. "
            "Train a checkpoint and point it to the local model directory."
        )
    if (os.path.isabs(MODEL_NAME) or MODEL_NAME.startswith(".")) and not Path(MODEL_NAME).is_dir():
        raise RuntimeError(f"Configured attribute classifier model directory does not exist: {MODEL_NAME}")
    model = SetFitModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    return model


model = _load_model()


def _attribute_text(attribute: AttributeValue) -> tuple[str, str, AttributeValue]:
    if isinstance(attribute, str):
        return attribute, attribute, attribute
    identifier = attribute.id or attribute.text
    return identifier, attribute.text, attribute


def classify_attributes(
    context: str,
    attributes: list[AttributeValue],
) -> AttributeClassificationResponse:
    logger.info("Classifying attributes: context=%r attributes=%r", context, attributes)
    prepared = [_attribute_text(attribute) for attribute in attributes]
    if not prepared:
        logger.info("Classification result: relevant_attributes=[]")
        return AttributeClassificationResponse()

    texts = [f"Context: {context}\nAttribute: {text}" for _, text, _ in prepared]
    scores: list[float] = []
    with MODEL_LOCK, torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            scores.extend(
                float(score)
                for score in model.predict_proba(texts[start : start + BATCH_SIZE])[:, 1].tolist()
            )

    results = [
        AttributeClassificationResult(
            id=identifier,
            attribute=original,
            score=score,
            relevant=score >= THRESHOLD,
        )
        for (identifier, _, original), score in zip(prepared, scores)
    ]
    response = AttributeClassificationResponse(
        relevant_attributes=[item.attribute for item in results if item.relevant],
        results=results,
    )
    logger.info("Classification result: relevant_attributes=%r", response.relevant_attributes)
    return response


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
            self._send_response(200, {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD})
            return
        self._send_response(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/classify":
            self._send_response(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = AttributeClassificationRequest.model_validate_json(self.rfile.read(length))
            response = classify_attributes(payload.context, payload.attributes)
            self._send_response(200, response.model_dump(mode="json"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_response(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - protects the service boundary
            self._send_response(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
