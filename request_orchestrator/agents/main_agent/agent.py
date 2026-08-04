from __future__ import annotations

from langsmith import traceable

from conversation.models.conversation_models import ConversationContext
from langgraph.graph import END, StateGraph
from personalization.profile.models import UserProfile
from request_orchestrator.agents.main_agent.request_analysis.analyze_request import analyze_request
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.agents.main_agent.validator.validator import validator
from request_orchestrator.constants import REQUEST_ANALYSIS_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.planner.planner import run_planner
from request_orchestrator.shared.synthesis.synthesis import run_synthesis
from typing import Any
from uuid import UUID


@traceable(name="Main Agent")
def run_agent(
    conversation_context: ConversationContext,
    user_query: str,
    conversation_id: str,
    roundtrip_id: str | None = None,
    max_turns: int = 10,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
) -> AgentResult:
    agentState = AgentState.new(
        task=user_query,
        max_turns=max_turns,
        conversation_context=conversation_context,
        user_profile=user_profile,
        roundtrip_id=UUID(roundtrip_id) if roundtrip_id else None,
        llm=llm,
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
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
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
