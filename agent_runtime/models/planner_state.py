from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.constants import NodeName
from agent_runtime.models.plan import Plan


@dataclass
class PlannerNodeState:
    node_name: str = NodeName.PLANNER.value
    plan: Plan | None = None
    needs_replan: bool = False
    plan_count: int = 0
    no_result_attempts: int = 0
    decision_status: str = ""
