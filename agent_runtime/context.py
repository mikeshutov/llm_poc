from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from agent_runtime.models.agent_state import AgentState

_current_agent_state: ContextVar[AgentState | None] = ContextVar("current_agent_state", default=None)


def get_current_conversation_id() -> str | None:
    state = get_current_agent_state()
    return state.execution_context.conversation_id if state is not None else None


def get_current_roundtrip_id() -> str | None:
    state = get_current_agent_state()
    roundtrip_id = state.execution_context.roundtrip_id if state is not None else None
    return str(roundtrip_id) if roundtrip_id is not None else None


def get_current_user_id() -> str | None:
    state = get_current_agent_state()
    return state.execution_context.user_profile.user_id if state is not None else None


def get_current_agent_name() -> str | None:
    state = get_current_agent_state()
    return state.agent_profile.name if state is not None else None


def get_current_agent_state() -> AgentState | None:
    return _current_agent_state.get()


@contextmanager
def bind_agent_context(agent_state: AgentState) -> Iterator[None]:
    token = _current_agent_state.set(agent_state)
    try:
        yield
    finally:
        _current_agent_state.reset(token)
