from __future__ import annotations

from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState
from request_orchestrator.shared.agents import load_agents
from request_orchestrator.shared.profile import load_user_profile
from request_orchestrator.shared.state_enrichment import enrich_state_node as shared_enrich_state_node
from request_orchestrator.shared.synthesis.synthesis import run_synthesis
from request_orchestrator.strategies.main_request_strategy import run_main_request_strategy


def _main_state_update(main_state: MainState) -> dict[str, MainState]:
    return {"main_state": main_state}


def load_agents_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    load_agents(main_state)
    return _main_state_update(main_state)


def enrich_state_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    return shared_enrich_state_node(state)


def run_main_request_strategy_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    return _main_state_update(run_main_request_strategy(state.main_state))


def load_user_profile_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    load_user_profile(main_state)
    return _main_state_update(main_state)


def run_synthesis_node(state: OrchestratorGraphState) -> dict[str, MainState]:
    main_state = state.main_state
    run_synthesis(main_state)
    return _main_state_update(main_state)
