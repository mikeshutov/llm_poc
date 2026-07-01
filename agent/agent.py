from __future__ import annotations

from langsmith import traceable

from agent.agentstate.model import AgentState
from conversation.models.conversation_models import ConversationContext
from agent.request_analysis.analyze_request import analyze_request
from agent.executor.executor import run_executor
from agent.graph_constants import REQUEST_ANALYSIS_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from agent.models.agent_result import AgentResult
from agent.planner.planner import run_planner
from agent.router.router import router
from agent.synthesis.synthesis import run_synthesis
from agent.validator.validator import validator
from langgraph.graph import END, StateGraph
from uuid import UUID


@traceable(name="Main Agent")
def run_agent(
    conversation_context: ConversationContext,
    user_query: str,
    conversation_id: str,
    roundtrip_id: str | None = None,
    max_turns: int = 10,
) -> AgentResult:
    agentState = AgentState.new(
        task=user_query,
        max_turns=max_turns,
        conversation_context=conversation_context,
        roundtrip_id=UUID(roundtrip_id) if roundtrip_id else None,
    )
    builder = StateGraph(AgentState)
    builder.add_node(REQUEST_ANALYSIS_EDGE, analyze_request)
    builder.add_node(PLAN_EDGE, run_planner)
    builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(REQUEST_ANALYSIS_EDGE)

    builder.add_conditional_edges(
        REQUEST_ANALYSIS_EDGE,
        router,
        {
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
            PLAN_EDGE: PLAN_EDGE,
        },
    )

    builder.add_conditional_edges(
        PLAN_EDGE,
        validator,
        {
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE},
    )

    builder.add_conditional_edges(
        EXECUTE_TOOLS_EDGE,
        router,
        {
            PLAN_EDGE: PLAN_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
    )

    builder.add_edge(SYNTHESIZE_EDGE, END)
    agent_graph = builder.compile()

    final_state = agent_graph.invoke(
        agentState,
        config={"configurable": {"thread_id": conversation_id}},
    )

    final = final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
    if final.result is None:
        raise ValueError("Agent finished without setting state.result")

    return final.result
