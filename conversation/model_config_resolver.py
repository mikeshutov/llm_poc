from __future__ import annotations

from conversation.models.conversation_model_config import ConversationModelConfig, ConversationModelConfigEntry


def resolve_conversation_model_config(
    entries: list[ConversationModelConfigEntry] | None = None,
) -> ConversationModelConfig:
    config = ConversationModelConfig.build_default()
    for entry in entries or []:
        config.set_value(entry.agent, entry.stage, entry.model)
    return config
