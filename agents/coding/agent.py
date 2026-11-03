from __future__ import annotations

from langsmith import traceable

from agent_runtime.models.agent_state import AgentState
from agent_runtime.runner import AgentRunner
from agent_runtime.strategies.react.graph import ReactStrategy
from agents.coding.profile import CODING_PROFILE


RUNNER = AgentRunner(CODING_PROFILE, ReactStrategy())


@traceable(name=CODING_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(agent_state)
