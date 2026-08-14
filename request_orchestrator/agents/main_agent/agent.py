from __future__ import annotations

from typing import Any
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from langsmith import traceable
from personalization.profile.models import UserProfile
from request_orchestrator.agent_stratagies.planner_executor_evaluator.graph import run_graph
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis

@traceable(name=MAIN_AGENT_PROFILE.name)
def run_agent(
    agent_state: AgentState | None = None,
    *,
    conversation_context: ConversationContext | None = None,
    user_query: str | None = None,
    conversation_id: str | None = None,
    roundtrip_id: str | None = None,
    max_turns: int = 10,
    user_profile: UserProfile | None = None,
    request_analysis: RequestAnalysis | None = None,
    llm: Any | None = None,
    conversation_model_config: ConversationModelConfig | None = None,
) -> AgentState:
    if agent_state is None:
        if conversation_context is None or user_query is None:
            raise ValueError("conversation_context and user_query are required when agent_state is not provided")
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
        if request_analysis is not None:
            agent_state.request_analysis = request_analysis.model_copy(deep=True)

    return run_graph(
        agent_state,
        execute_router=router,
        thread_id=agent_state.conversation_id or conversation_id or "",
    )
