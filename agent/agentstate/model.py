import os
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.models import AgentResult, Plan


def flatten_conversation_entries(entries: list[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in entries
        if m.get("content")
    )


class ClassificationResults(BaseModel):
    applicable_tool_categories: list[str] = []
    can_answer_without_tools: bool = False
    confidence: float = 0.0


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
    conversation_entries: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    classification_results: ClassificationResults = field(default_factory=ClassificationResults)
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    goal_reached: bool = False
    llm: Any = field(
        default_factory=lambda: ChatOpenAI(model=os.getenv("AGENT_MODEL", "gpt-4.1"), temperature=0)
    )

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        conversation_entries: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_entries=[] if conversation_entries is None else conversation_entries,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration
