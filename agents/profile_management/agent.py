from __future__ import annotations

from langsmith import traceable

from agent_runtime.runner import AgentRunner
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from agent_runtime.models.agent_state import AgentState
from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy

RUNNER = AgentRunner(
    PROFILE_MANAGEMENT_PROFILE,
    DirectReturnStrategy(),
)


@traceable(name=PROFILE_MANAGEMENT_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(
        agent_state,
    )
