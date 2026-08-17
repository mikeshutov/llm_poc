from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module


class AgentRegistry:
    @lru_cache
    def get(self, agent_name: str) -> Callable:
        module = import_module(f"request_orchestrator.agents.{agent_name}")
        runner = getattr(module, "run_agent", None)
        if not callable(runner):
            raise AttributeError(f"Agent module {module.__name__!r} does not expose a callable 'run_agent'")
        return runner


agent_registry = AgentRegistry()
