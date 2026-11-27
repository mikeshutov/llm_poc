from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from common.logging import record_node_log
from agent_runtime.constants import NodeName
from agent_runtime.models.main_state import MainState
from agent_runtime.models.orchestrator_graph_state import OrchestratorGraphState
from capabilities.discovery.attribute_discovery import attribute_discovery
from capabilities.discovery.capability_discovery import capability_discovery
from agents.main.profile_loading import load_user_profile
from agents.main.request import initialize_request_agent_state

def enrich_state(
    main_state: MainState,
    *,
    include_agents: bool = True,
    attribute_lookup=attribute_discovery,
    capability_lookup=capability_discovery,
) -> MainState:
    """Run independent state enrichment lookups for the current request."""
    started_at = perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="state-enrichment") as executor:
            attribute_future = executor.submit(attribute_lookup, main_state)
            capability_state = initialize_request_agent_state(main_state)
            capability_state.available_agent_states = main_state.agent_states
            capability_future = executor.submit(
                capability_lookup,
                capability_state,
                include_agents=include_agents,
                query=main_state.discovery_query or None,
                query_embedding=main_state.discovery_query_embedding,
                allowed_tool_names=capability_state.agent_profile.tool_names,
            )
            main_state.discovered_attribute_types = attribute_future.result()
            main_state.discovered_capabilities = capability_future.result()
            capability_state.inputs.discovered_capabilities = main_state.discovered_capabilities
            main_state.agent_states = capability_state.available_agent_states or {}
            load_user_profile(main_state)
    except Exception as exc:
        record_node_log(
            node_name=NodeName.ENRICH_STATE,
            stage="enrichment",
            event_type="state_enrichment",
            level="error",
            message="State enrichment failed",
            roundtrip_id=main_state.execution_context.roundtrip_id,
            payload={"error": str(exc)},
            duration_ms=int((perf_counter() - started_at) * 1000),
        )
        raise
    record_node_log(
        node_name=NodeName.ENRICH_STATE,
        stage="enrichment",
        event_type="state_enrichment",
        message="State enrichment completed",
        roundtrip_id=main_state.execution_context.roundtrip_id,
        payload={
            "attribute_count": len(main_state.discovered_attribute_types),
            "tool_count": len(main_state.discovered_capabilities.tools) if main_state.discovered_capabilities else 0,
            "agent_count": len(main_state.discovered_capabilities.agents) if main_state.discovered_capabilities else 0,
        },
        duration_ms=int((perf_counter() - started_at) * 1000),
    )
    return main_state


def enrich_state_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    return {"main_state": enrich_state(state.main_state)}
