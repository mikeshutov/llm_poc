from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.agents.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis, RequestAnalysisGoal


class AgentStratagy(Protocol):
    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState: ...


@dataclass(frozen=True)
class AgentRunner:
    profile: AgentProfile
    stratagy: AgentStratagy

    def _build_default_request_analysis(self) -> RequestAnalysis:
        goal = self.profile.request_analysis_goal.strip()
        if not goal:
            return RequestAnalysis()
        return RequestAnalysis(
            goals=[
                RequestAnalysisGoal(
                    agent=self.profile.name,
                    goal=goal,
                    tool_categories=sorted(self.profile.allowed_categories),
                )
            ],
        )

    def _prepare_state(self, agent_state: AgentState) -> AgentState:
        agent_state.agent_profile = self.profile
        agent_state.max_turns = self.profile.max_turns
        if not agent_state.request_analysis.goal_for_agent(self.profile.name) and self.profile.request_analysis_goal.strip():
            agent_state.request_analysis = self._build_default_request_analysis()
        return agent_state

    def run(
        self,
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
        if agent_state is None:
            if conversation_context is None or user_query is None:
                raise ValueError("conversation_context and user_query are required when agent_state is not provided")
            agent_state = AgentState.new(
                task=user_query,
                max_turns=self.profile.max_turns if max_turns is None else max_turns,
                conversation_context=conversation_context,
                user_profile=user_profile,
                agent_profile=self.profile,
                conversation_id=conversation_id,
                roundtrip_id=UUID(roundtrip_id) if roundtrip_id else None,
                llm=llm,
                conversation_model_config=conversation_model_config,
            )
            if request_analysis is not None:
                agent_state.request_analysis = request_analysis.model_copy(deep=True)

        prepared_state = self._prepare_state(agent_state)
        return self.stratagy.run(
            prepared_state,
            thread_id=prepared_state.conversation_id or conversation_id or "",
        )
