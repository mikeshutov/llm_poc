from __future__ import annotations

from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_state import AgentState
from agent_runtime.runner import AgentRunner, AgentStrategy


def execute_agent(
    agent_profile: AgentProfile,
    strategy: AgentStrategy,
    agent_state: AgentState,
) -> AgentState:
    """Execute a prepared delegated agent state with its selected strategy."""
    return AgentRunner(profile=agent_profile, strategy=strategy).run(agent_state)


__all__ = ["execute_agent"]
