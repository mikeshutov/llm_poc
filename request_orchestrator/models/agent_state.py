from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from conversation.models.conversation_model_config import (
    ConversationModelConfig,
    PLANNER_STAGE,
)
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import GeoLocation, GeoMetadata, UserProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_profile import AgentProfile
from .agent_result import AgentResult
from .evaluation_result import EvaluationStatus, EVALUATION_STATUS_RETRYABLE
from .plan import Plan


class RequestAnalysisGoal(BaseModel):
    agent: str = ""
    goal: str = ""
    tool_categories: list[str] = []


class RequestAnalysis(BaseModel):
    goals: list[RequestAnalysisGoal] = []
    requested_user_attribute_types: list[str] = []

    def _goal_entry_for_agent(self, agent_name: str) -> RequestAnalysisGoal | None:
        normalized_agent_name = agent_name.strip()
        for goal_entry in self.goals:
            if goal_entry.agent.strip() == normalized_agent_name:
                return goal_entry
        return None

    def goal_for_agent(self, agent_name: str, default: str = "") -> str:
        goal_entry = self._goal_entry_for_agent(agent_name)
        if goal_entry is None:
            return default
        return goal_entry.goal.strip()

    def tool_categories_for_agent(self, agent_name: str) -> list[str]:
        goal_entry = self._goal_entry_for_agent(agent_name)
        if goal_entry is None:
            return []
        return [
            category
            for category in goal_entry.tool_categories
            if isinstance(category, str) and category.strip()
        ]

    def set_goal_for_agent(self, agent_name: str, goal: str, *, tool_categories: list[str] | None = None) -> None:
        goal_entry = self._goal_entry_for_agent(agent_name)
        normalized_agent_name = agent_name.strip()
        normalized_goal = goal.strip()
        normalized_tool_categories = [] if tool_categories is None else [
            category.strip()
            for category in tool_categories
            if isinstance(category, str) and category.strip()
        ]
        if goal_entry is not None:
            goal_entry.goal = normalized_goal
            if tool_categories is not None:
                goal_entry.tool_categories = normalized_tool_categories
            return
        self.goals.append(
            RequestAnalysisGoal(
                agent=normalized_agent_name,
                goal=normalized_goal,
                tool_categories=normalized_tool_categories,
            )
        )

def _default_planner_model_for_agent_scope(agent_scope: str) -> str:
    return ConversationModelConfig.build_default().resolve(
        agent=agent_scope,
        stage=PLANNER_STAGE,
    )

def _build_default_llm() -> ChatOpenAI:
    return ChatOpenAI(model=ConversationModelConfig.default_main_agent_planner_model())


@dataclass
class IterationState:
    plan: Plan | None = None
    results: dict[str, Any] = field(default_factory=dict)
    needs_replan: bool = False

    @classmethod
    def new(
        cls,
        *,
        plan: Plan | None = None,
        results: dict[str, Any] | None = None,
    ) -> "IterationState":
        return cls(
            plan=plan,
            results={} if results is None else results,
            needs_replan=False,
        )

    def clone(self) -> IterationState:
        return IterationState(
            plan=None if self.plan is None else self.plan.model_copy(deep=True),
            results=dict(self.results),
            needs_replan=self.needs_replan,
        )


@dataclass
class AgentState:
    task: str
    max_turns: int
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    agent_profile: AgentProfile = field(default_factory=lambda: MAIN_AGENT_PROFILE)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    relevant_evidence_ids: list[str] = field(default_factory=list)
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    goal_reached: bool = False
    llm: Any = field(default_factory=_build_default_llm, repr=False)
    conversation_model_config: ConversationModelConfig = field(default_factory=ConversationModelConfig.build_default)

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        agent_profile: AgentProfile,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
    ) -> "AgentState":
        resolved_conversation_model_config = (
            ConversationModelConfig.build_default()
            if conversation_model_config is None
            else conversation_model_config
        )
        resolved_agent_scope = agent_profile.scope
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            agent_profile=agent_profile,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
            llm=(
                ChatOpenAI(
                    model=resolved_conversation_model_config.resolve(
                        agent=resolved_agent_scope,
                        stage=PLANNER_STAGE,
                    )
                )
                if llm is None
                else llm
            ),
            conversation_model_config=resolved_conversation_model_config,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration

    def clone_for_parallel(self) -> AgentState:
        return AgentState(
            task=self.task,
            max_turns=self.max_turns,
            conversation_context=self.conversation_context,
            user_profile=self.user_profile,
            agent_profile=self.agent_profile,
            conversation_id=self.conversation_id,
            roundtrip_id=self.roundtrip_id,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            result=self.result,
            relevant_evidence_ids=list(self.relevant_evidence_ids),
            evaluation_status=self.evaluation_status,
            goal_reached=self.goal_reached,
            llm=self.llm,
            conversation_model_config=self.conversation_model_config.model_copy(deep=True),
        )

    def resolve_model_for_stage(self, *, agent: str, stage: str) -> str:
        return self.conversation_model_config.resolve(agent, stage)

    def resolve_agent_scope(self) -> str:
        return self.agent_profile.scope

    def build_llm_for_stage(self, *, stage: str, agent: str | None = None) -> Any:
        resolved_agent = self.resolve_agent_scope() if agent is None else agent
        if self.llm is None:
            return ChatOpenAI(
                model=self.conversation_model_config.resolve(
                    agent=resolved_agent,
                    stage=stage,
                )
            )
        if not isinstance(self.llm, ChatOpenAI):
            return self.llm
        model_name = self.resolve_model_for_stage(agent=resolved_agent, stage=stage)
        agent_scope = self.resolve_agent_scope()
        default_planner_model = _default_planner_model_for_agent_scope(agent_scope)
        if resolved_agent == agent_scope and stage == PLANNER_STAGE and model_name == default_planner_model:
            return self.llm
        return ChatOpenAI(model=model_name)



