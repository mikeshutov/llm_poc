from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

from conversation.models.conversation_model_config import ConversationModelConfig

_current_conversation_id: ContextVar[str | None] = ContextVar("current_conversation_id", default=None)
_current_roundtrip_id: ContextVar[str | None] = ContextVar("current_roundtrip_id", default=None)
_current_conversation_model_config: ContextVar[ConversationModelConfig | None] = ContextVar("current_conversation_model_config", default=None)


def get_current_conversation_id() -> str | None:
    return _current_conversation_id.get()


def get_current_roundtrip_id() -> str | None:
    return _current_roundtrip_id.get()


def get_current_conversation_model_config() -> ConversationModelConfig | None:
    return _current_conversation_model_config.get()


@contextmanager
def bind_runtime_context(
    *,
    conversation_id: str | None,
    conversation_model_config: ConversationModelConfig | None,
    roundtrip_id: str | None = None,
) -> Iterator[None]:
    conversation_token: Token[str | None] = _current_conversation_id.set(conversation_id)
    roundtrip_token: Token[str | None] = _current_roundtrip_id.set(roundtrip_id)
    config_token: Token[ConversationModelConfig | None] = _current_conversation_model_config.set(conversation_model_config)
    try:
        yield
    finally:
        _current_conversation_id.reset(conversation_token)
        _current_roundtrip_id.reset(roundtrip_token)
        _current_conversation_model_config.reset(config_token)
