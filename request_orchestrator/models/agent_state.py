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
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.plan import Plan


class RequestAnalysis(BaseModel):
    goal: str = ""
    applicable_tool_categories: list[str] = []
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
    # map evidence id -> tool output (or normalized string)
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


@dataclass
class AgentState:
    task: str
    max_turns: int
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    goal_reached: bool = False
    llm: Any = field(default_factory=lambda: ChatOpenAI(model=LLM_MODEL))

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration
