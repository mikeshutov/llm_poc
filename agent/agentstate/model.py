from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.models import AgentResult, Plan
from conversation.models.conversation_models import ConversationContext
from common.model_constants import LLM_MODEL


class RequestAnalysis(BaseModel):
    goal: str = ""
    applicable_tool_categories: list[str] = []
    requires_tools: bool = False
    context_answer_confidence: float = 0.0


class GeoMetadata(BaseModel):
    current_datetime: str
    current_date: str
    current_weekday: str
    timezone: str


def build_geometadata(*, timezone: str = "America/Toronto") -> GeoMetadata:
    now = datetime.now(ZoneInfo(timezone))
    return GeoMetadata(
        current_datetime=now.isoformat(),
        current_date=now.date().isoformat(),
        current_weekday=now.strftime("%A"),
        timezone=timezone,
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
    geometadata: GeoMetadata = field(default_factory=build_geometadata)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    goal_reached: bool = False
    llm: Any = field(
        default_factory=lambda: ChatOpenAI(model=LLM_MODEL)
    )

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        conversation_context: ConversationContext | None = None,
        geometadata: GeoMetadata | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            geometadata=build_geometadata() if geometadata is None else geometadata,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration
