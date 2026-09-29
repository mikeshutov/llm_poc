from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from llm.clients.embeddings import embed_text
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.discovery.models import (
    AvailableAgent,
    AvailableTool,
    CapabilityDiscoveryResult,
)
from tool.tools import TOOL_CATEGORIES

MAX_DISCOVERED_CAPABILITIES = 5
RETRIEVAL_K = max(
    MAX_DISCOVERED_CAPABILITIES,
    int(os.getenv("CAPABILITY_DISCOVERY_RETRIEVAL_TOP_K", "20")),
)
TOP_K = min(
    MAX_DISCOVERED_CAPABILITIES,
    max(1, int(os.getenv("CAPABILITY_DISCOVERY_TOP_K", str(MAX_DISCOVERED_CAPABILITIES)))),
)
MIN_SIMILARITY = float(os.getenv("CAPABILITY_DISCOVERY_MIN_SIMILARITY", "0.35"))
SEMANTIC_WEIGHT = float(os.getenv("CAPABILITY_DISCOVERY_SEMANTIC_WEIGHT", "1.0"))
BM25_WEIGHT = float(os.getenv("CAPABILITY_DISCOVERY_BM25_WEIGHT", "0.7"))
RRF_K = max(1, int(os.getenv("CAPABILITY_DISCOVERY_RRF_K", "60")))
BM25_K1 = float(os.getenv("CAPABILITY_DISCOVERY_BM25_K1", "1.2"))
BM25_B = float(os.getenv("CAPABILITY_DISCOVERY_BM25_B", "0.75"))
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)


@lru_cache(maxsize=256)
def _cached_embed_text(text: str, embedder_key: int) -> tuple[float, ...]:
    return tuple(embed_text(text))


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _bm25_rank(query: str, candidates: list[tuple[str, str, Any]]) -> list[tuple[float, int, str]]:
    """Rank candidates lexically using BM25."""
    query_terms = _tokenize(query)
    if not query_terms or not candidates:
        return []
    documents = [(candidate_id, _tokenize(text)) for candidate_id, text, _ in candidates]
    document_frequency: dict[str, int] = {}
    for _, terms in documents:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    average_length = sum(len(terms) for _, terms in documents) / len(documents)
    if average_length == 0:
        return []

    query_frequency = {term: query_terms.count(term) for term in set(query_terms)}
    scored: list[tuple[float, int, str]] = []
    for index, (candidate_id, terms) in enumerate(documents):
        term_frequency = {term: terms.count(term) for term in query_frequency}
        normalization = 1 - BM25_B + BM25_B * len(terms) / average_length
        score = 0.0
        for term, qtf in query_frequency.items():
            tf = term_frequency[term]
            if not tf:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            score += idf * (tf * (BM25_K1 + 1) / (tf + BM25_K1 * normalization)) * qtf
        if score > 0:
            scored.append((score, index, candidate_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _semantic_rank(
    query_embedding: list[float] | None, candidates: list[tuple[str, str, Any]]
) -> list[tuple[float, int, str]]:
    if query_embedding is None:
        return []
    ranked: list[tuple[float, int, str]] = []
    for index, (candidate_id, searchable_text, _) in enumerate(candidates):
        score = _similarity(
            query_embedding,
            list(_cached_embed_text(searchable_text, id(embed_text))),
        )
        if score >= MIN_SIMILARITY:
            ranked.append((score, index, candidate_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[:RETRIEVAL_K]


def _rrf_rank(
    candidate_ids: list[str],
    ranked_lists: list[tuple[float, list[tuple[float, int, str]]]],
) -> list[str]:
    """Fuse ranked lists with weighted reciprocal rank fusion."""
    original_order = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    scores: dict[str, float] = {}
    for weight, ranked in ranked_lists:
        if weight <= 0:
            continue
        for rank, (_, _, candidate_id) in enumerate(ranked, start=1):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (RRF_K + rank)
    return sorted(scores, key=lambda candidate_id: (-scores[candidate_id], original_order[candidate_id]))


def _rank_candidates(
    query: str,
    query_embedding: list[float] | None,
    candidates: list[tuple[str, str, Any]],
) -> list[tuple[float, Any]]:
    lexical_rank = _bm25_rank(query, candidates)
    semantic_rank = _semantic_rank(query_embedding, candidates) if query_embedding is not None else []
    ranked_ids = _rrf_rank(
        [candidate_id for candidate_id, _, _ in candidates],
        [(SEMANTIC_WEIGHT, semantic_rank), (BM25_WEIGHT, lexical_rank)],
    )
    values = {candidate_id: value for candidate_id, _, value in candidates}
    return [(0.0, values[candidate_id]) for candidate_id in ranked_ids[:TOP_K]]


def _build_tool_candidates() -> list[tuple[str, str, AvailableTool]]:
    candidates: list[tuple[str, str, AvailableTool]] = []
    for category in TOOL_CATEGORIES.values():
        for tool in sorted(category.tools, key=lambda candidate: candidate.name):
            available_tool = AvailableTool(
                name=tool.name,
                description=tool.description,
            )
            candidates.append(
                (
                    tool.name,
                    f"{tool.name}: {tool.description}",
                    available_tool,
                )
            )
    return candidates


def _build_agent_candidates(main_state: MainState) -> list[tuple[str, str, AvailableAgent]]:
    candidates: list[tuple[str, str, AvailableAgent]] = []
    for agent_state in main_state.agent_states.values():
        profile = agent_state.agent_profile
        if not profile.delegatable:
            continue
        available_agent = AvailableAgent(
            agent=profile.name,
            description=profile.description,
            tools=sorted(profile.tool_names),
        )
        searchable_text = " ".join(
            [
                profile.name,
                profile.description,
                " ".join(profile.tool_names),
            ]
        )
        candidates.append((profile.name, searchable_text, available_agent))
    return candidates


def capability_discovery(
    main_state: MainState,
    *,
    include_agents: bool = True,
) -> CapabilityDiscoveryResult:
    """Find concrete capabilities relevant to the current request.

    Agent discovery is only enabled for the top-level request. Agent planner
    iterations discover tools and attributes, but cannot delegate to agents.
    """
    from request_orchestrator.shared.agents.agent_selection_query import build_agent_selection_query

    query = build_agent_selection_query(
        current_user_request=main_state.task,
        conversation_context=main_state.execution_context.conversation_context,
    )
    tool_candidates = _build_tool_candidates()
    agent_candidates = _build_agent_candidates(main_state) if include_agents else []
    try:
        query_embedding: list[float] | None = embed_text(query)
    except Exception:
        query_embedding = None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="capability-discovery") as executor:
        tool_future = executor.submit(_rank_candidates, query, query_embedding, tool_candidates)
        agent_future = executor.submit(_rank_candidates, query, query_embedding, agent_candidates)
        tool_matches = tool_future.result()
        agent_matches = agent_future.result()

    # Preserve a useful discovery result for a query with no lexical or
    # semantic hit, while applying TOP_K to the fallback.
    if not tool_matches:
        tool_matches = [(0.0, value) for _, _, value in tool_candidates[:TOP_K]]
    if not agent_matches:
        agent_matches = [(0.0, value) for _, _, value in agent_candidates[:TOP_K]]

    return CapabilityDiscoveryResult(
        agents=[value for _, value in agent_matches],
        tools=[value for _, value in tool_matches],
    )
