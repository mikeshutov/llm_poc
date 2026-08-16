from __future__ import annotations

from typing import Any

from langsmith import traceable
from request_orchestrator.agent_runner import AgentRunner
from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.main_agent.router.router import router
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.request_analysis import RequestAnalysis

RUNNER = AgentRunner(
    MAIN_AGENT_PROFILE,
    PlannerExecutorEvaluatorStratagy(router),
)


@traceable(name=MAIN_AGENT_PROFILE.name)
def run_agent(
    agent_state: AgentState | None = None,
    *,
    user_query: str | None = None,
    execution_context: AgentExecutionContext | None = None,
    request_analysis: RequestAnalysis | None = None,
    llm: Any | None = None,
) -> AgentState:
    return RUNNER.run(
        agent_state,
        user_query=user_query,
        execution_context=execution_context,
        request_analysis=request_analysis,
        llm=llm,
    )
