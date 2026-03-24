from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.agentstate.model import AgentState
from agents.executor.executor import run_executor
from agents.graph_constants import EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from agents.impl.general.planner.planner import run_planner
from agents.impl.general.router.router import router
from agents.impl.general.validator.validator import validator


def general_agent_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node(PLAN_EDGE, run_planner)
    builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
    builder.set_entry_point(PLAN_EDGE)

    builder.add_conditional_edges(
        PLAN_EDGE,
        validator,
        {
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
            SYNTHESIZE_EDGE: END,
        },
    )

    builder.add_conditional_edges(
        EXECUTE_TOOLS_EDGE,
        router,
        {
            PLAN_EDGE: PLAN_EDGE,
            SYNTHESIZE_EDGE: END,
        },
    )

    return builder.compile()
