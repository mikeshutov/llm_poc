from __future__ import annotations

from typing import Any
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from langgraph.graph import END, StateGraph
from langsmith import traceable
from personalization.profile.models import UserProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.request_analysis.analyze_request import analyze_request
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.agents.main_agent.validator.validator import validator
from request_orchestrator.agents.profile_management import run_agent as run_profile_management_agent
from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, REQUEST_ANALYSIS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.evaluator import evaluator_router, run_evaluator
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.planner.planner import run_planner
from request_orchestrator.shared.profile import load_user_profile
from request_orchestrator.shared.synthesis.synthesis import run_synthesis

COLLECT_EDGE = "collect"
FANOUT_EDGE = "fanout"
INITIAL_PLAN_EDGE = "initial_plan"
PROFILE_LOADING_EDGE = "load_user_profile"
PROFILE_MANAGEMENT_EDGE = "profile_management"


def _fanout_node(state: AgentState) -> AgentState:
    return state


def _collect_node(state: AgentState) -> AgentState:
    return state


def _run_planner_update(state: AgentState) -> dict[str, object]:
    planner_state = state.clone_for_parallel()
    updated_state = run_planner(planner_state)
    return {
        "iteration_trace": updated_state.iteration_trace,
        "goal_reached": updated_state.goal_reached,
        "result": updated_state.result,
        "agent_log": updated_state.agent_log.clone(),
    }


def _run_profile_management_update(state: AgentState) -> dict[str, object]:
    profile_state = state.clone_for_parallel()
    updated_state = run_profile_management_agent(profile_state)
    return {
        "subagent_states": {
            name: subagent_state.clone()
            for name, subagent_state in updated_state.subagent_states.items()
        },
    }


@traceable(name=MAIN_AGENT_PROFILE.name)
def run_agent(
    conversation_context: ConversationContext,
    user_query: str,
    conversation_id: str,
    roundtrip_id: str | None = None,
    max_turns: int = 10,
    user_profile: UserProfile | None = None,
    llm: Any | None = None,
    conversation_model_config: ConversationModelConfig | None = None,
) -> AgentResult:
    agent_state = AgentState.new(
        task=user_query,
        max_turns=max_turns,
        conversation_context=conversation_context,
        user_profile=user_profile,
        agent_profile=MAIN_AGENT_PROFILE,
        conversation_id=conversation_id,
        roundtrip_id=UUID(roundtrip_id) if roundtrip_id else None,
        llm=llm,
        conversation_model_config=conversation_model_config,
    )

    builder = StateGraph(AgentState)
    builder.add_node(REQUEST_ANALYSIS_EDGE, analyze_request)
    builder.add_node(PROFILE_LOADING_EDGE, load_user_profile)
    builder.add_node(FANOUT_EDGE, _fanout_node)
    builder.add_node(PROFILE_MANAGEMENT_EDGE, _run_profile_management_update)
    builder.add_node(INITIAL_PLAN_EDGE, _run_planner_update)
    builder.add_node(COLLECT_EDGE, _collect_node)
    builder.add_node(PLAN_EDGE, _run_planner_update)
    builder.add_node(EVALUATE_EDGE, run_evaluator)
    builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(REQUEST_ANALYSIS_EDGE)

    builder.add_edge(REQUEST_ANALYSIS_EDGE, PROFILE_LOADING_EDGE)
    builder.add_edge(PROFILE_LOADING_EDGE, FANOUT_EDGE)
    builder.add_edge(FANOUT_EDGE, PROFILE_MANAGEMENT_EDGE)
    builder.add_edge(FANOUT_EDGE, INITIAL_PLAN_EDGE)
    builder.add_edge(PROFILE_MANAGEMENT_EDGE, COLLECT_EDGE)
    builder.add_edge(INITIAL_PLAN_EDGE, COLLECT_EDGE)

    builder.add_conditional_edges(
        COLLECT_EDGE,
        validator,
        {
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
    )

    builder.add_edge(PLAN_EDGE, COLLECT_EDGE)

    builder.add_conditional_edges(
        EVALUATE_EDGE,
        evaluator_router,
        {
            PLAN_EDGE: PLAN_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
    )

    builder.add_conditional_edges(
        EXECUTE_TOOLS_EDGE,
        router,
        {
            PLAN_EDGE: PLAN_EDGE,
            EVALUATE_EDGE: EVALUATE_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
    )

    builder.add_edge(SYNTHESIZE_EDGE, END)
    agent_graph = builder.compile()

    final_state = agent_graph.invoke(
        agent_state,
        config={"configurable": {"thread_id": conversation_id}},
    )

    final = final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
    if final.result is None:
        raise ValueError("Agent finished without setting state.result")

    return final.result
