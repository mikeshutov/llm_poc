import os
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from agent.models import AgentResult, Plan

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
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_entries=[] if conversation_entries is None else conversation_entries,
            conversation_id=conversation_id,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration
