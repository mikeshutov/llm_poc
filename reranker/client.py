from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Protocol

import requests
from pydantic import ValidationError

from reranker.models import (
    RerankerCandidate,
    RerankerRequest,
    RerankerRequestCandidate,
    RerankerResponse,
    RerankerScore,
)

RERANKER_CANDIDATE_TOKEN_BUDGET = max(64, int(os.getenv("RERANKER_CANDIDATE_TOKEN_BUDGET", "384")))


class RerankerUnavailableError(RuntimeError):
    pass


class RerankerBackend(Protocol):
    def rerank(self, query: str, candidates: list[RerankerCandidate]) -> list[RerankerScore]: ...


@dataclass(frozen=True)
class RerankerClient:
    base_url: str = os.getenv("RERANKER_SERVICE_URL", "http://localhost:5433")
    timeout_seconds: float = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "15"))
    batch_size: int = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "8")))

    def prepare_candidates(
        self,
        candidates: list[RerankerCandidate],
        *,
        token_budget: int | None = None,
    ) -> list[RerankerRequestCandidate]:
        max_chars = max(64, token_budget or RERANKER_CANDIDATE_TOKEN_BUDGET) * 4
        return [
            RerankerRequestCandidate(
                id=candidate.id,
                text=self._serialize_fields(candidate.fields, max_chars=max_chars),
            )
            for candidate in candidates
        ]

    @staticmethod
    def _serialize_fields(fields: dict[str, Any], *, max_chars: int) -> str:
        lines: list[str] = []
        for name, value in fields.items():
            if value is None or value == "" or value == [] or value == {}:
                continue
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=True, default=str)
            lines.append(f"{name}: {value}")
        return "\n".join(lines)[:max_chars].rstrip()

    def rerank(self, query: str, candidates: list[RerankerCandidate]) -> list[RerankerScore]:
        candidates = self.prepare_candidates(candidates)
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/rerank",
                json=RerankerRequest(query=query, candidates=candidates).model_dump(mode="json"),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = RerankerResponse.model_validate(response.json())
        except (requests.RequestException, ValueError, ValidationError) as exc:
            raise RerankerUnavailableError("The local reranker service is unavailable") from exc
        return payload.results
