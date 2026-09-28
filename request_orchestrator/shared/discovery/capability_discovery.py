from __future__ import annotations

import math
import os
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
def _cached_embed_text(text: str) -> tuple[float, ...]:
    return tuple(embed_text(text))


def _rank_candidates(
    query_embedding: list[float],
    candidates: list[tuple[str, str, Any]],
) -> list[tuple[float, Any]]:
    ranked: list[tuple[float, int, Any]] = []
    for index, (_, searchable_text, value) in enumerate(candidates):
        score = _similarity(query_embedding, list(_cached_embed_text(searchable_text)))
        if score >= MIN_SIMILARITY:
            ranked.append((score, index, value))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    retrieved = ranked[:RETRIEVAL_K]

    # This is the post-retrieval ranking boundary. A future dedicated
    # reranker can operate on `retrieved` without changing the public result
    # size exposed to planners.
    retrieved.sort(key=lambda item: (-item[0], item[1]))
    return [(score, value) for score, _, value in retrieved[:TOP_K]]


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
        query_embedding = embed_text(query)
    except Exception:
        return CapabilityDiscoveryResult(
            agents=[value for _, _, value in agent_candidates],
            tools=[value for _, _, value in tool_candidates],
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="capability-discovery") as executor:
        tool_future = executor.submit(_rank_candidates, query_embedding, tool_candidates)
        agent_future = executor.submit(_rank_candidates, query_embedding, agent_candidates)
        tool_matches = tool_future.result()
        agent_matches = agent_future.result()

    discovered_tools = [value for _, value in tool_matches]
    return CapabilityDiscoveryResult(
        agents=[value for _, value in agent_matches],
        tools=discovered_tools,
    )
