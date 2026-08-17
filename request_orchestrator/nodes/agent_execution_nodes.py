from __future__ import annotations

from langgraph.types import Send

from request_orchestrator.agents.registry import agent_registry
from request_orchestrator.constants import APPLY_AGENT_UPDATES_EDGE, RUN_SINGLE_AGENT_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState


def fanout_agent_runs_node(state: OrchestratorGraphState) -> list[Send] | str:
    main_state = state.main_state
    sends = [
        Send(
            RUN_SINGLE_AGENT_EDGE,
            main_state.agent_states[profile.name],
        )
        for profile in main_state.agent_profiles
        if main_state.agent_states[profile.name].inputs.task.strip()
    ]
    if not sends:
        return APPLY_AGENT_UPDATES_EDGE
    return sends


def run_single_agent_node(state: AgentState) -> dict[str, dict[str, AgentState]]:
    runner = agent_registry.get(state.agent_name)
    updated_agent_state = runner(state)
    return {
        "completed_agents": {
            state.agent_name: updated_agent_state,
        }
    }
