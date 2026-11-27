from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from agent_runtime.models.agent_state import AgentState

current_agent_state: ContextVar[AgentState | None] = ContextVar("current_agent_state", default=None)


@contextmanager
def bind_agent_context(agent_state: AgentState) -> Iterator[None]:
    token = current_agent_state.set(agent_state)
    try:
        yield
    finally:
        current_agent_state.reset(token)
