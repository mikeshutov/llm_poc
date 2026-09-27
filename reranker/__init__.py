from typing import TYPE_CHECKING

from reranker.client import RerankerClient, RerankerUnavailableError
from reranker.query import build_reranker_query
from reranker.models import (
    Candidate,
    RerankerRequest,
    RerankerRequestCandidate,
    RerankerResponse,
    RerankerScore,
)
from reranker.constants import DEFAULT_TOP_K

if TYPE_CHECKING:
    from reranker.service import CandidateReranker, rerank_candidates


# TODO: Remove lazy service exports once the conversation/evidence/reranker
# dependency cycle is resolved at the module boundary.
def __getattr__(name: str):
    if name in {"CandidateReranker", "rerank_candidates"}:
        from reranker.service import CandidateReranker, rerank_candidates

        return {
            "CandidateReranker": CandidateReranker,
            "rerank_candidates": rerank_candidates,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Candidate",
    "RerankerRequest",
    "RerankerRequestCandidate",
    "RerankerResponse",
    "RerankerScore",
    "RerankerClient",
    "RerankerUnavailableError",
    "build_reranker_query",
    "CandidateReranker",
    "rerank_candidates",
    "DEFAULT_TOP_K",
]
