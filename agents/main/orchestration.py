from __future__ import annotations

from agent_runtime.models.main_state import MainState
from agent_runtime.models.orchestrator_graph_state import OrchestratorGraphState
from agents.discovery import load_agents
from agents.main.profile_loading import load_user_profile
from agent_runtime.state_enrichment import enrich_state_node as shared_enrich_state_node
from agents.main.request import run_main_request_strategy


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


__all__ = [
    "enrich_state_node",
    "load_agents_node",
    "load_user_profile_node",
    "run_main_request_strategy_node",
]
