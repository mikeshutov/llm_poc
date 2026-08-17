from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable

from request_orchestrator.constants import (
    APPLY_AGENT_UPDATES_EDGE,
    DISTRIBUTE_GOALS_EDGE,
    PROFILE_LOADING_EDGE,
    REQUEST_ANALYSIS_EDGE,
    RUN_SINGLE_AGENT_EDGE,
    SYNTHESIZE_EDGE,
)
from request_orchestrator.models.orchestrator_graph_state import OrchestratorGraphState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.nodes.agent_execution_nodes import fanout_agent_runs_node, run_single_agent_node
from request_orchestrator.nodes.main_state_nodes import (
    apply_agent_updates_node,
    distribute_goals_node,
    load_user_profile_node,
    run_request_analysis_node,
    run_synthesis_node,
)


class OrchestratorGraph:
    def __init__(self) -> None:
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(OrchestratorGraphState)
        builder.add_node(REQUEST_ANALYSIS_EDGE, run_request_analysis_node)
        builder.add_node(PROFILE_LOADING_EDGE, load_user_profile_node)
        builder.add_node(DISTRIBUTE_GOALS_EDGE, distribute_goals_node)
        builder.add_node(RUN_SINGLE_AGENT_EDGE, run_single_agent_node)
        builder.add_node(APPLY_AGENT_UPDATES_EDGE, apply_agent_updates_node)
        builder.add_node(SYNTHESIZE_EDGE, run_synthesis_node)
        builder.set_entry_point(REQUEST_ANALYSIS_EDGE)

        builder.add_edge(REQUEST_ANALYSIS_EDGE, PROFILE_LOADING_EDGE)
        builder.add_edge(PROFILE_LOADING_EDGE, DISTRIBUTE_GOALS_EDGE)
        builder.add_conditional_edges(DISTRIBUTE_GOALS_EDGE, fanout_agent_runs_node)
        builder.add_edge(RUN_SINGLE_AGENT_EDGE, APPLY_AGENT_UPDATES_EDGE)
        builder.add_edge(APPLY_AGENT_UPDATES_EDGE, SYNTHESIZE_EDGE)
        builder.add_edge(SYNTHESIZE_EDGE, END)
        
        return builder.compile()

    def run(self, main_state: MainState) -> OrchestratorResult:
        final_state = self._graph.invoke(
            OrchestratorGraphState(main_state=main_state),
            config={"configurable": {"thread_id": main_state.execution_context.conversation_id or ""}},
        )
        if isinstance(final_state, OrchestratorGraphState):
            return final_state.main_state.result.copy()
        if isinstance(final_state, dict):
            return final_state.main_state.result.copy()
        return final_state.main_state.result.copy()


_ORCHESTRATOR = OrchestratorGraph()


@traceable(name="request_orchestrator")
def run_agent(main_state: MainState) -> OrchestratorResult:
    return _ORCHESTRATOR.run(main_state)
