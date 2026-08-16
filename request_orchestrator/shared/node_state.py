from __future__ import annotations

from dataclasses import dataclass, field

from request_orchestrator.shared.evaluator_state import EvaluatorNodeState
from request_orchestrator.shared.planner_state import PlannerNodeState


@dataclass
class NodeState:
    node_name: str


@dataclass
class AgentNodeStates:
    planner: PlannerNodeState = field(default_factory=PlannerNodeState)
    evaluator: EvaluatorNodeState = field(default_factory=EvaluatorNodeState)
