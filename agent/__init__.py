from __future__ import annotations

from typing import Any


def run_agent(*args: Any, **kwargs: Any):
    from agent.agent import run_agent as _run_agent

    return _run_agent(*args, **kwargs)

__all__ = ["run_agent"]
