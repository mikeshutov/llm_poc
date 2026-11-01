from __future__ import annotations

from langsmith import traceable

from agent_runtime.models.agent_state import AgentState
from agent_runtime.runner import AgentRunner
from agent_runtime.strategies.react.graph import ReactStrategy
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE


RUNNER = AgentRunner(
    DOCUMENT_WRITING_PROFILE,
    ReactStrategy(),
)


@traceable(name=DOCUMENT_WRITING_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(agent_state)
