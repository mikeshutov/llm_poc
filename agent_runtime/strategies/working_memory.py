from __future__ import annotations

from pathlib import Path

from common.config import UTF8_ENCODING
from agent_runtime.models.agent_profile import AgentExecutionStrategy


_STRATEGY_TEMPLATE_PATHS = {
    AgentExecutionStrategy.REACT: Path(__file__).with_name("react") / "working-memory.md",
    AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR: (
        Path(__file__).with_name("plan_execute_evaluate") / "working-memory.md"
    ),
    AgentExecutionStrategy.DIRECT_RETURN: Path(__file__).with_name("single_pass") / "working-memory.md",
}


def load_strategy_working_memory_template(strategy: AgentExecutionStrategy) -> str:
    """Load the default runtime-memory template owned by an execution strategy."""
    path = _STRATEGY_TEMPLATE_PATHS.get(strategy)
    if path is None:
        return ""
    return path.read_text(encoding=UTF8_ENCODING).strip()


__all__ = ["load_strategy_working_memory_template"]
