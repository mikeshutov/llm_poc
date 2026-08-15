from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile


@dataclass
class AgentExecutionContext:
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    model_config: ConversationModelConfig = field(default_factory=ConversationModelConfig.build_default)
