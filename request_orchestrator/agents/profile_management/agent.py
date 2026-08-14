from __future__ import annotations

from langsmith import traceable

from request_orchestrator.agent_stratagies.planner_executor_evaluator.graph import run_graph
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.agents.profile_management.router import router
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis


@traceable(name=PROFILE_MANAGEMENT_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    agent_state.max_turns = agent_state.agent_profile.max_turns
    agent_state.request_analysis = RequestAnalysis(
        goal=agent_state.agent_profile.request_analysis_goal,
        applicable_tool_categories=sorted(agent_state.agent_profile.allowed_categories),
    )
    return run_graph(
        agent_state,
        execute_router=router,
        thread_id=agent_state.conversation_id or "",
    )
