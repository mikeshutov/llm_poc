from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState


def merge_completed_agents(
    current: dict[str, AgentState],
    new: dict[str, AgentState],
) -> dict[str, AgentState]:
    return current | new


@dataclass
class OrchestratorGraphState:
    main_state: MainState
    completed_agents: Annotated[dict[str, AgentState], merge_completed_agents] = field(default_factory=dict)
