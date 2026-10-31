from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from common.logging import record_node_log
from agent_runtime.constants import NodeName
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from agent_runtime.models.orchestrator_graph_state import OrchestratorGraphState
from capabilities.discovery.attribute_discovery import attribute_discovery
from capabilities.discovery.capability_discovery import capability_discovery
from agents.main.profile_loading import load_user_profile

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
            capability_state = main_state
            if include_agents:
                capability_future = executor.submit(capability_lookup, capability_state)
            else:
                capability_future = executor.submit(
                    capability_lookup,
                    capability_state,
                    include_agents=False,
                )
            main_state.discovered_attribute_types = attribute_future.result()
            main_state.discovered_capabilities = capability_future.result()
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


def discover_agent_state(agent_state: AgentState) -> AgentState:
    """Refresh discovery only when execution feedback starts a new plan."""
    iteration_task = agent_state.inputs.task
    missing_information = agent_state.node_states.evaluator.missing_information
    if missing_information:
        iteration_task += "\n\nUnresolved information:\n" + "\n".join(
            f"- {item}" for item in missing_information
        )
    if agent_state.node_states.planner.no_result_attempts:
        iteration_task += "\n\nThe previous execution produced no useful result; choose a materially different capability."

    discovery_state = MainState(
        task=iteration_task,
        execution_context=agent_state.execution_context,
        agent_profiles=[agent_state.agent_profile],
        agent_states={agent_state.agent_profile.name: agent_state},
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-discovery") as executor:
        attribute_future = executor.submit(attribute_discovery, discovery_state)
        capability_future = executor.submit(
            capability_discovery,
            discovery_state,
            include_agents=False,
        )
        discovered_attribute_types = attribute_future.result()
        agent_state.inputs.discovered_attribute_types = discovered_attribute_types
        agent_state.inputs.discovered_capabilities = capability_future.result()

    if discovered_attribute_types:
        load_user_profile(discovery_state, title="Additional User Attributes Loaded")
    return agent_state


def discover_agent_state_node(agent_state: AgentState) -> AgentState:
    return discover_agent_state(agent_state)
