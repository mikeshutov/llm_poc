from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from agent_runtime.runner import AgentRunner
from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentKind,
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
        if agent_profile.kind == AgentKind.USER_AGENT:
            return self._build_runner(agent_profile)
        module = import_module(f"agents.{agent_profile.name}")
        runner = getattr(module, "run_agent", None)
        if not callable(runner):
            raise AttributeError(f"Agent module {module.__name__!r} does not expose a callable 'run_agent'")
        return runner

    def _build_runner(self, agent_profile: AgentProfile) -> Callable:
        strategy = self._strategies.get(agent_profile.execution_strategy)
        if strategy is None:
            raise KeyError(f"No runner strategy registered for {agent_profile.execution_strategy!r}")
        runner = AgentRunner(agent_profile, strategy)
        return runner.run


agent_registry = AgentRegistry()
