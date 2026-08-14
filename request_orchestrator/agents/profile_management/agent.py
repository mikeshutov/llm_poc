from __future__ import annotations

from langsmith import traceable

from request_orchestrator.agent_runner import AgentRunner
from request_orchestrator.agent_stratagies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.agents.profile_management.router import router
from request_orchestrator.models.agent_state import AgentState

RUNNER = AgentRunner(
    PROFILE_MANAGEMENT_PROFILE,
    PlannerExecutorEvaluatorStratagy(router),
)


@traceable(name=PROFILE_MANAGEMENT_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(
        agent_state,
    )
