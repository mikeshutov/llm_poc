from __future__ import annotations

from time import perf_counter

from langsmith import traceable
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log

from llm.clients.embeddings import embed_text
from agents.repository.repo_factory import get_agent_repo
from agent_runtime.models.main_state import MainState
from agents.discovery.agent_selection_query import build_agent_selection_query
from agent_runtime.constants import NodeName


@traceable(name="Load Agents Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def load_agents(main_state: MainState) -> MainState:
    started_at = perf_counter()
    main_state.discovery_query = build_agent_selection_query(
        current_user_request=main_state.task,
        conversation_context=main_state.execution_context.conversation_context,
    )
    user_id = (main_state.execution_context.user_profile.user_id or "").strip()
    if not user_id:
        record_node_log(
            node_name=NodeName.LOAD_AGENTS,
            stage="discovery",
            event_type="agent_discovery",
            message="Agent discovery skipped because no user was selected",
            payload={"query": main_state.discovery_query, "agent_count": 0, "skipped": "missing_user_id"},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return main_state

    task = main_state.task.strip()
    if not task:
        record_node_log(
            node_name=NodeName.LOAD_AGENTS,
            stage="discovery",
            event_type="agent_discovery",
            message="Agent discovery skipped because the request was empty",
            payload={"query": main_state.discovery_query, "agent_count": 0, "skipped": "empty_task"},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return main_state

    agent_repository = get_agent_repo()
    if not agent_repository.list_for_user(user_id):
        record_node_log(
            node_name=NodeName.LOAD_AGENTS,
            stage="discovery",
            event_type="agent_discovery",
            message="No user agents were available for discovery",
            payload={"query": main_state.discovery_query, "agent_count": 0},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return main_state

    try:
        main_state.discovery_query_embedding = embed_text(main_state.discovery_query)
    except Exception as exc:
        main_state.discovery_query_embedding = None
        record_node_log(
            node_name=NodeName.LOAD_AGENTS,
            stage="embedding",
            event_type="agent_discovery",
            level="warning",
            message="Agent discovery embedding failed",
            payload={"query": main_state.discovery_query, "error": str(exc), "agent_count": 0},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return main_state
    agents = agent_repository.list_relevant_for_user(
        user_id,
        query_embedding=main_state.discovery_query_embedding,
    )
    loaded_profiles = [agent.to_agent_profile() for agent in agents]
    if not loaded_profiles:
        record_node_log(
            node_name=NodeName.LOAD_AGENTS,
            stage="discovery",
            event_type="agent_discovery",
            message="Agent discovery returned no matching agents",
            payload={"query": main_state.discovery_query, "agent_count": 0},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        return main_state

    existing_names = {profile.name for profile in main_state.agent_profiles}
    for profile in loaded_profiles:
        if profile.name in existing_names:
            continue
        main_state.agent_profiles.append(profile)
        existing_names.add(profile.name)
    main_state.initialize_agent_states()
    record_node_log(
        node_name=NodeName.LOAD_AGENTS,
        stage="discovery",
        event_type="agent_discovery",
        message="Agent discovery completed",
        payload={
            "query": main_state.discovery_query,
            "agent_count": len(loaded_profiles),
            "agents": [profile.name for profile in loaded_profiles],
        },
        duration_ms=int((perf_counter() - started_at) * 1000),
    )
    return main_state
