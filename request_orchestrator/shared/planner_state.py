from __future__ import annotations

from dataclasses import dataclass

from request_orchestrator.models.plan import Plan


@dataclass
class PlannerNodeState:
    node_name: str = "planner"
    plan: Plan | None = None
    needs_replan: bool = False
    plan_count: int = 0
