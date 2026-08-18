from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from request_orchestrator.agent_runner import AgentRunner
from request_orchestrator.agent_runner.models.agent_profile import AgentKind, AgentProfile
from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.agents.main_agent.router.router import router


class AgentRegistry:
    def __init__(self) -> None:
        self._dynamic_strategy = PlannerExecutorEvaluatorStratagy(router)

    def get(self, agent_profile: AgentProfile) -> Callable:
        if agent_profile.kind == AgentKind.USER_AGENT:
            return self._build_runner(agent_profile)
        module = import_module(f"request_orchestrator.agents.{agent_profile.name}")
        runner = getattr(module, "run_agent", None)
        if not callable(runner):
            raise AttributeError(f"Agent module {module.__name__!r} does not expose a callable 'run_agent'")
        return runner

    def _build_runner(self, agent_profile: AgentProfile) -> Callable:
        runner = AgentRunner(agent_profile, self._dynamic_strategy)
        return runner.run


agent_registry = AgentRegistry()
