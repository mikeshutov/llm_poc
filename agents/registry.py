from __future__ import annotations

from collections.abc import Callable
from langsmith import traceable

from agent_runtime.runner import AgentRunner
from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
)
from agent_runtime.strategies.plan_execute_evaluate.graph import PlannerExecutorEvaluatorStrategy
from agent_runtime.strategies.react.graph import ReactStrategy
from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy
from agent_runtime.routing.main_router import router


class AgentRegistry:
    def __init__(self) -> None:
        self._strategies = {
            AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR: PlannerExecutorEvaluatorStrategy(router),
            AgentExecutionStrategy.REACT: ReactStrategy(),
            AgentExecutionStrategy.DIRECT_RETURN: DirectReturnStrategy(),
        }

    def get(self, agent_profile: AgentProfile) -> Callable:
        strategy = self._strategies.get(agent_profile.execution_strategy)
        if strategy is None:
            raise KeyError(f"No runner strategy registered for {agent_profile.execution_strategy!r}")
        runner = AgentRunner(agent_profile, strategy)
        return traceable(name=agent_profile.name)(runner.run)


agent_registry = AgentRegistry()
