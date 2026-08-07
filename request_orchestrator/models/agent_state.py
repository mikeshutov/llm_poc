from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from common.model_constants import LLM_MODEL
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import GeoLocation, GeoMetadata, UserProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.plan import Plan


class RequestAnalysis(BaseModel):
    goal: str = ""
    applicable_tool_categories: list[str] = []
    requested_user_attribute_types: list[str] = []
    requires_tools: bool = False
    context_answer_confidence: float = 0.0


def build_geometadata(
    *,
    timezone: str | None = "America/Toronto",
    location: GeoLocation | None = None,
) -> GeoMetadata:
    resolved_timezone = (timezone or "").strip()
    if not resolved_timezone and location is not None:
        resolved_timezone = (location.timezone or "").strip()
    if not resolved_timezone:
        resolved_timezone = "America/Toronto"

    now = datetime.now(ZoneInfo(resolved_timezone))
    return GeoMetadata(
        current_datetime=now.isoformat(),
        current_date=now.date().isoformat(),
        current_weekday=now.strftime("%A"),
        timezone=resolved_timezone,
        location=location,
    )


@dataclass
class IterationState:
    plan: Plan | None = None
    results: dict[str, Any] = field(default_factory=dict)

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
        )

    def clone(self) -> IterationState:
        return IterationState(
            plan=None if self.plan is None else self.plan.model_copy(deep=True),
            results=dict(self.results),
        )


@dataclass
class SubagentState:
    task: str
    max_turns: int
    agent_profile: AgentProfile
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    goal_reached: bool = False

    def to_runtime_state(self, parent_state: AgentState) -> AgentState:
        return AgentState(
            task=self.task,
            max_turns=self.max_turns,
            conversation_context=parent_state.conversation_context,
            user_profile=parent_state.user_profile,
            agent_profile=self.agent_profile,
            conversation_id=parent_state.conversation_id,
            roundtrip_id=parent_state.roundtrip_id,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            result=self.result,
            goal_reached=self.goal_reached,
            llm=parent_state.llm,
        )

    def update_from_runtime_state(self, runtime_state: AgentState) -> SubagentState:
        self.task = runtime_state.task
        self.max_turns = runtime_state.max_turns
        self.agent_profile = runtime_state.agent_profile
        self.request_analysis = runtime_state.request_analysis.model_copy(deep=True)
        self.iteration_trace = [iteration.clone() for iteration in runtime_state.iteration_trace]
        self.result = runtime_state.result
        self.goal_reached = runtime_state.goal_reached
        return self

    def clone(self) -> SubagentState:
        return SubagentState(
            task=self.task,
            max_turns=self.max_turns,
            agent_profile=self.agent_profile,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            result=self.result,
            goal_reached=self.goal_reached,
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
    subagent_states: dict[str, SubagentState] = field(default_factory=dict)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    goal_reached: bool = False
    llm: Any = field(default_factory=lambda: ChatOpenAI(model=LLM_MODEL), repr=False)

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        agent_profile: AgentProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            agent_profile=MAIN_AGENT_PROFILE if agent_profile is None else agent_profile,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
            llm=ChatOpenAI(model=LLM_MODEL) if llm is None else llm,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration

    def get_subagent_state(
        self,
        agent_profile: AgentProfile,
        *,
        task: str | None = None,
        max_turns: int | None = None,
    ) -> SubagentState:
        existing_state = self.subagent_states.get(agent_profile.name)
        if existing_state is not None:
            if task is not None:
                existing_state.task = task
            if max_turns is not None:
                existing_state.max_turns = max_turns
            existing_state.agent_profile = agent_profile
            return existing_state

        subagent_state = SubagentState(
            task=self.task if task is None else task,
            max_turns=self.max_turns if max_turns is None else max_turns,
            agent_profile=agent_profile,
        )
        self.subagent_states[agent_profile.name] = subagent_state
        return subagent_state

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
            subagent_states={name: subagent_state.clone() for name, subagent_state in self.subagent_states.items()},
            result=self.result,
            goal_reached=self.goal_reached,
            llm=self.llm,
        )
