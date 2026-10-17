from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from request_orchestrator.models.plan import Plan


@dataclass
class PlannerNodeState:
    node_name: str = "planner"
    plan: Plan | None = None
    tool_calls: list[Any] = field(default_factory=list)
    needs_replan: bool = False
    plan_count: int = 0
    no_result_attempts: int = 0
