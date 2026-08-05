from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable

from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.agents.profile_management.router import router
from request_orchestrator.constants import EXECUTE_TOOLS_EDGE, PLAN_EDGE
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis, SubagentState
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.planner.planner import run_planner

PROFILE_MANAGEMENT_ROUTER_EDGE = "profile_management_router"
MAX_PROFILE_MANAGEMENT_TURNS = 5


def _route_node(state: AgentState) -> AgentState:
    return state


def _build_profile_management_task() -> str:
    return (
        "Review this turn for durable user attribute maintenance needs. "
        "If attribute work is needed, plan the minimal retrieval and/or update step combination required."
    )


def _prepare_subagent_state(parent_state: AgentState) -> SubagentState:
    subagent_task = _build_profile_management_task()
    subagent_state = parent_state.get_subagent_state(
        PROFILE_MANAGEMENT_PROFILE,
        task=subagent_task,
        max_turns=MAX_PROFILE_MANAGEMENT_TURNS,
    )
    subagent_state.request_analysis = RequestAnalysis(
        goal=subagent_task,
        applicable_tool_categories=sorted(PROFILE_MANAGEMENT_PROFILE.allowed_categories),
        requires_tools=True,
        context_answer_confidence=0.0,
    )
    return subagent_state


def _compile_graph():
    builder = StateGraph(AgentState)
    builder.add_node(PROFILE_MANAGEMENT_ROUTER_EDGE, _route_node)
    builder.add_node(PLAN_EDGE, run_planner)
    builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
    builder.set_entry_point(PROFILE_MANAGEMENT_ROUTER_EDGE)

    builder.add_conditional_edges(
        PROFILE_MANAGEMENT_ROUTER_EDGE,
        router,
        {
            PLAN_EDGE: PLAN_EDGE,
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
            END: END,
        },
    )

    builder.add_edge(PLAN_EDGE, PROFILE_MANAGEMENT_ROUTER_EDGE)
    builder.add_edge(EXECUTE_TOOLS_EDGE, PROFILE_MANAGEMENT_ROUTER_EDGE)

    return builder.compile()


@traceable(name=PROFILE_MANAGEMENT_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    agent_graph = _compile_graph()
    subagent_state = _prepare_subagent_state(agent_state)
    runtime_state = subagent_state.to_runtime_state(agent_state)

    final_state = agent_graph.invoke(
        runtime_state,
        config={"configurable": {"thread_id": runtime_state.conversation_id or ""}},
    )

    final_runtime_state = final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
    subagent_state.update_from_runtime_state(final_runtime_state)
    agent_state.subagent_states[PROFILE_MANAGEMENT_PROFILE.name] = subagent_state
    return agent_state
