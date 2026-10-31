from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable
from common.logging import compact_trace_inputs, compact_trace_outputs

from agent_runtime.constants import (
    ENRICH_STATE_EDGE,
    LOAD_AGENTS_EDGE,
    PROFILE_LOADING_EDGE,
)
from agent_runtime.models.orchestrator_graph_state import OrchestratorGraphState
from agent_runtime.models.main_state import MainState
from agent_runtime.models.orchestrator_result import OrchestratorResult
from agents.main.orchestration import (
    enrich_state_node,
    load_agents_node,
    load_user_profile_node,
    run_main_request_strategy_node,
)


class OrchestratorGraph:
    def __init__(self) -> None:
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(OrchestratorGraphState)
        builder.add_node(LOAD_AGENTS_EDGE, load_agents_node)
        builder.add_node(ENRICH_STATE_EDGE, enrich_state_node)
        builder.add_node(PROFILE_LOADING_EDGE, load_user_profile_node)
        builder.add_node("main_request_strategy", run_main_request_strategy_node)
        builder.set_entry_point(LOAD_AGENTS_EDGE)

        builder.add_edge(LOAD_AGENTS_EDGE, ENRICH_STATE_EDGE)
        builder.add_edge(ENRICH_STATE_EDGE, PROFILE_LOADING_EDGE)
        builder.add_edge(PROFILE_LOADING_EDGE, "main_request_strategy")
        builder.add_edge("main_request_strategy", END)
        
        return builder.compile()

    def run(self, main_state: MainState) -> OrchestratorResult:
        final_state = self._graph.invoke(
            OrchestratorGraphState(main_state=main_state),
            config={"configurable": {"thread_id": main_state.execution_context.conversation_id or ""}},
        )
        if isinstance(final_state, OrchestratorGraphState):
            return final_state.main_state.result.copy()
        if isinstance(final_state, dict):
            return final_state["main_state"].result.copy()
        return final_state.main_state.result.copy()


_ORCHESTRATOR = OrchestratorGraph()


@traceable(name="request_orchestrator", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_agent(main_state: MainState) -> OrchestratorResult:
    return _ORCHESTRATOR.run(main_state)
