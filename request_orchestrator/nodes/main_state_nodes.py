from __future__ import annotations

from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState
from request_orchestrator.shared.agents import load_user_agents
from request_orchestrator.shared.profile import load_user_profile
from request_orchestrator.shared.request_analysis.analyze_request import analyze_request
from request_orchestrator.shared.synthesis.synthesis import run_synthesis


def _main_state_update(main_state: MainState) -> dict[str, MainState]:
    return {"main_state": main_state}


def load_user_agents_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    load_user_agents(main_state)
    return _main_state_update(main_state)


def run_request_analysis_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    analyze_request(main_state)
    return _main_state_update(main_state)


def load_user_profile_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    load_user_profile(main_state)
    return _main_state_update(main_state)


def distribute_goals_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    main_state.initialize_agent_states()
    return _main_state_update(main_state)


def apply_agent_updates_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    main_state.agent_states.update(state.completed_agents)
    return _main_state_update(main_state)


def run_synthesis_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    run_synthesis(main_state)
    return _main_state_update(main_state)
