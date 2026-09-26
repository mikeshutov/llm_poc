from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import requests


class RerankerUnavailableError(RuntimeError):
    pass


class RerankerBackend(Protocol):
    def rerank(self, query: str, candidates: list[dict[str, str]]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class RerankerClient:
    base_url: str = os.getenv("RERANKER_SERVICE_URL", "http://localhost:5433")
    timeout_seconds: float = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "15"))
    batch_size: int = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "8")))

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/rerank",
                json={"query": query, "candidates": candidates},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RerankerUnavailableError("The local reranker service is unavailable") from exc

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RerankerUnavailableError("The reranker service returned an invalid response")
        normalized: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("id"), str):
                raise RerankerUnavailableError("The reranker service returned an invalid result")
            try:
                score = float(result["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankerUnavailableError("The reranker service returned an invalid score") from exc
            normalized.append({"id": result["id"], "score": score})
        return normalized
