from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langsmith import traceable

from request_orchestrator.agents.main_agent.agent import run_agent as run_main_agent
from request_orchestrator.agents.profile_management.agent import run_agent as run_profile_management_agent
from request_orchestrator.constants import REQUEST_ANALYSIS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.profile import load_user_profile
from request_orchestrator.shared.request_analysis.analyze_request import analyze_request
from request_orchestrator.shared.synthesis.synthesis import run_synthesis

PROFILE_MANAGEMENT_EDGE = "profile_management"
MAIN_AGENT_EDGE = "main_agent"
COLLECT_EDGE = "collect"
PROFILE_LOADING_EDGE = "load_user_profile"
DISTRIBUTE_GOALS_EDGE = "distribute_goals"
AGENT_RUNNERS: dict[str, Callable] = {
    PROFILE_MANAGEMENT_EDGE: run_profile_management_agent,
    MAIN_AGENT_EDGE: run_main_agent,
}
AGENT_EXECUTION_ORDER = [
    PROFILE_MANAGEMENT_EDGE,
    MAIN_AGENT_EDGE,
]


def _should_run_agent(state: MainState, agent_name: str) -> bool:
    return bool(state.request_analysis.goal_for_agent(agent_name))


def _distribute_goals_node(state: MainState) -> MainState:
    state.distribute_goals_to_agent_states()
    return state


def _build_run_agent_node(agent_name: str, runner: Callable) -> Callable[[MainState], MainState]:
    def _run_agent_update(state: MainState) -> MainState:
        if not _should_run_agent(state, agent_name):
            return state
        agent_state = state.get_agent_state(agent_name).clone_for_parallel()
        updated_state = runner(agent_state)
        state.upsert_agent_state(updated_state)
        return state

    return _run_agent_update


def _collect_node(state: MainState) -> MainState:
    state.collect_agent_outputs()
    return state


def _compile_graph():
    builder = StateGraph(MainState)
    builder.add_node(REQUEST_ANALYSIS_EDGE, analyze_request)
    builder.add_node(PROFILE_LOADING_EDGE, load_user_profile)
    builder.add_node(DISTRIBUTE_GOALS_EDGE, _distribute_goals_node)
    for agent_name in AGENT_EXECUTION_ORDER:
        builder.add_node(agent_name, _build_run_agent_node(agent_name, AGENT_RUNNERS[agent_name]))
    builder.add_node(COLLECT_EDGE, _collect_node)
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(REQUEST_ANALYSIS_EDGE)

    builder.add_edge(REQUEST_ANALYSIS_EDGE, PROFILE_LOADING_EDGE)
    builder.add_edge(PROFILE_LOADING_EDGE, DISTRIBUTE_GOALS_EDGE)
    previous_edge = DISTRIBUTE_GOALS_EDGE
    for agent_name in AGENT_EXECUTION_ORDER:
        builder.add_edge(previous_edge, agent_name)
        previous_edge = agent_name
    builder.add_edge(previous_edge, COLLECT_EDGE)
    builder.add_edge(COLLECT_EDGE, SYNTHESIZE_EDGE)
    builder.add_edge(SYNTHESIZE_EDGE, END)
    return builder.compile()


@traceable(name="request_orchestrator")
def run_agent(main_state: MainState) -> AgentResult:
    graph = _compile_graph()
    final_state = graph.invoke(
        main_state,
        config={"configurable": {"thread_id": main_state.conversation_id or ""}},
    )
    final = final_state if isinstance(final_state, MainState) else MainState(**final_state)
    return final.build_final_result()
