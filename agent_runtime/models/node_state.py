from __future__ import annotations

from dataclasses import dataclass, field

from agent_runtime.models.evaluator_state import EvaluatorNodeState
from agent_runtime.models.planner_state import PlannerNodeState


@dataclass
class NodeState:
    node_name: str


@dataclass
class AgentNodeStates:
    planner: PlannerNodeState = field(default_factory=PlannerNodeState)
    evaluator: EvaluatorNodeState = field(default_factory=EvaluatorNodeState)
