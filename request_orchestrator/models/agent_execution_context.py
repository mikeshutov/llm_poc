from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from conversation.models.conversation_models import ConversationContext
from llm.conversation_model_config import ConversationModelConfig
from personalization.profile.models import UserProfile


@dataclass
class AgentExecutionContext:
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    model_config: ConversationModelConfig = field(default_factory=ConversationModelConfig.build_default)

    @classmethod
    def new(
        cls,
        *,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        model_config: ConversationModelConfig | None = None,
    ) -> "AgentExecutionContext":
        return cls(
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            model_config=ConversationModelConfig.build_default() if model_config is None else model_config,
        )
