from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState
from request_orchestrator.shared.discovery.attribute_discovery import attribute_discovery
from request_orchestrator.shared.discovery.capability_discovery import capability_discovery
from request_orchestrator.shared.profile.load_user_profile import load_user_profile


def enrich_state(
    main_state: MainState,
    *,
    include_agents: bool = True,
    attribute_lookup=attribute_discovery,
    capability_lookup=capability_discovery,
) -> MainState:
    """Run independent state enrichment lookups for the current request."""
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
    return main_state


def enrich_state_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    return {"main_state": enrich_state(state.main_state)}


def enrich_agent_state(
    agent_state: AgentState,
    *,
    attribute_lookup=attribute_discovery,
    capability_lookup=capability_discovery,
    profile_loader=load_user_profile,
) -> AgentState:
    """Recompute tools and relevant attributes for one planner iteration."""
    iteration_task_parts = [agent_state.inputs.task]
    missing_information = agent_state.node_states.evaluator.missing_information
    if missing_information:
        iteration_task_parts.append(
            "Unresolved information from the previous iteration:\n"
            + "\n".join(f"- {item}" for item in missing_information)
        )
    if agent_state.node_states.planner.no_result_attempts:
        iteration_task_parts.append(
            "The previous tool execution produced no useful result. Find a materially different capability."
        )
    iteration_state = MainState(
        task="\n\n".join(part for part in iteration_task_parts if part.strip()),
        execution_context=agent_state.execution_context,
        agent_profiles=(
            [agent_state.agent_profile]
            if agent_state.available_agent_states is None
            else [candidate.agent_profile for candidate in agent_state.available_agent_states.values()]
        ),
        agent_states=(
            {agent_state.agent_profile.name: agent_state}
            if agent_state.available_agent_states is None
            else agent_state.available_agent_states
        ),
    )
    attribute_state = MainState(
        task=iteration_state.task,
        execution_context=agent_state.execution_context,
        agent_profiles=[agent_state.agent_profile],
        agent_states={agent_state.agent_profile.name: agent_state},
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-state-enrichment") as executor:
        attribute_future = executor.submit(attribute_lookup, attribute_state)
        capability_future = executor.submit(
            capability_lookup,
            iteration_state,
            include_agents=agent_state.available_agent_states is not None,
        )
        discovered_attribute_types = attribute_future.result()
        attribute_state.discovered_attribute_types = discovered_attribute_types
        agent_state.inputs.discovered_attribute_types = discovered_attribute_types
        agent_state.inputs.discovered_capabilities = capability_future.result()
    if discovered_attribute_types:
        profile_loader(attribute_state)
    return agent_state


def enrich_agent_state_node(agent_state: AgentState) -> AgentState:
    return enrich_agent_state(agent_state)
