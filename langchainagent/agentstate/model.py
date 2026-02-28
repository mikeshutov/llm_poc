from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class PlanStep:
    thought: str
    ref: str        
    tool: str    
    args: str     

@dataclass
class IterationState:
    plan_string: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    steps: list[PlanStep] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        plan_string: str = "",
        results: dict[str, Any] | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> "IterationState":
        return cls(
            plan_string=plan_string,
            results={} if results is None else results,
            steps=[] if steps is None else steps,
        )


@dataclass
class AgentState:
    task: str
    max_turns: int
    iteration_trace: list[IterationState] = field(default_factory=list)
    result: str = ""
    goal_reached : bool = False

    @classmethod
    def new(cls, task: str, max_turns: int) -> "AgentState":
        return cls(task=task, max_turns = max_turns)

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration
