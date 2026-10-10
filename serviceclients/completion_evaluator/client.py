from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from pydantic import ValidationError

from serviceclients.completion_evaluator.models import CompletionEvaluatorRequest, CompletionEvaluatorResult


class CompletionEvaluatorUnavailableError(RuntimeError):
    """Raised when the local completion evaluator cannot be reached or parsed."""


@dataclass(frozen=True)
class CompletionEvaluatorClient:
    base_url: str = os.getenv("COMPLETION_EVALUATOR_SERVICE_URL", "http://localhost:5435")
    timeout_seconds: float = float(os.getenv("COMPLETION_EVALUATOR_TIMEOUT_SECONDS", "15"))

    def evaluate(self, premise: str, hypothesis: str) -> CompletionEvaluatorResult:
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/evaluate",
                json=CompletionEvaluatorRequest(premise=premise, hypothesis=hypothesis).model_dump(mode="json"),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return CompletionEvaluatorResult.model_validate(response.json())
        except (requests.RequestException, ValueError, ValidationError) as exc:
            raise CompletionEvaluatorUnavailableError(
                f"The local completion evaluator service is unavailable at {self.base_url}: {exc}"
            ) from exc
