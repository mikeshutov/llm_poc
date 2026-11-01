from __future__ import annotations

from langsmith import traceable

from agent_runtime.models.agent_state import AgentState
from agent_runtime.runner import AgentRunner
from agent_runtime.routing.main_router import router
from agent_runtime.strategies.plan_execute_evaluate.graph import (
    PlannerExecutorEvaluatorStrategy,
)
from agents.research.profile import RESEARCH_PROFILE


RUNNER = AgentRunner(
    RESEARCH_PROFILE,
    PlannerExecutorEvaluatorStrategy(router),
)


@traceable(name=RESEARCH_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(agent_state)
