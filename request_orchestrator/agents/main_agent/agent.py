from __future__ import annotations

from langsmith import traceable
from request_orchestrator.agent_runner import AgentRunner
from request_orchestrator.agent_stratagies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.models.agent_state import AgentState

RUNNER = AgentRunner(
    MAIN_AGENT_PROFILE,
    PlannerExecutorEvaluatorStratagy(router),
)


@traceable(name=MAIN_AGENT_PROFILE.name)
def run_agent(agent_state: AgentState) -> AgentState:
    return RUNNER.run(
        agent_state,
    )
