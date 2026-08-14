from __future__ import annotations

from typing import Any

from langsmith import traceable
from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.agent_runner import AgentRunner
from request_orchestrator.agent_stratagies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis

RUNNER = AgentRunner(
    MAIN_AGENT_PROFILE,
    PlannerExecutorEvaluatorStratagy(router),
)


@traceable(name=MAIN_AGENT_PROFILE.name)
def run_agent(
    agent_state: AgentState | None = None,
    *,
    conversation_context: ConversationContext | None = None,
    user_query: str | None = None,
    conversation_id: str | None = None,
    roundtrip_id: str | None = None,
    max_turns: int | None = None,
    user_profile: UserProfile | None = None,
    request_analysis: RequestAnalysis | None = None,
    llm: Any | None = None,
    conversation_model_config: ConversationModelConfig | None = None,
) -> AgentState:
    return RUNNER.run(
        agent_state,
        conversation_context=conversation_context,
        user_query=user_query,
        conversation_id=conversation_id,
        roundtrip_id=roundtrip_id,
        max_turns=max_turns,
        user_profile=user_profile,
        request_analysis=request_analysis,
        llm=llm,
        conversation_model_config=conversation_model_config,
    )
