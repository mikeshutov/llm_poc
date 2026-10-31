from __future__ import annotations

from llm.conversation_model_config import PROFILE_AGENT_MODEL_SCOPE
from personalization.profile.models import UserProfile
from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
    PROFILE_MANAGEMENT_AGENT_NAME,
)
from capabilities.adapters.profile import (
    set_user_display_name,
    set_user_first_name,
    set_user_last_name,
    update_user_tone,
)
from capabilities.adapters.user_attributes import upsert_user_attribute
from agents.profile_management.prompts import build_profile_management_rules


def build_profile_management_profile(user_profile: UserProfile | None = None) -> AgentProfile:
    static_tools = [set_user_display_name, update_user_tone, upsert_user_attribute]
    if user_profile is None or not (user_profile.first_name or '').strip():
        static_tools.append(set_user_first_name)
    if user_profile is None or not (user_profile.last_name or '').strip():
        static_tools.append(set_user_last_name)

    return AgentProfile(
        name=PROFILE_MANAGEMENT_AGENT_NAME,
        scope=PROFILE_AGENT_MODEL_SCOPE,
        version=3,
        description=(
            "Maintain durable user profile fields and attributes. Use this whenever "
            "the user states or hints at preferences, avoidances, interests, or other durable "
            "personal details that should update their user attributes."
        ),
        allowed_categories=set(),
        static_tools=static_tools,
        execution_strategy=AgentExecutionStrategy.DIRECT_RETURN,
        delegatable=True,
        max_turns=5,
        planner_instruction=(
            'Maintain durable user profile fields and durable user attributes. '
            'Do not answer the user. '
            'If no profile mutation is needed, return no steps.'
        ),
        planner_rules=build_profile_management_rules(),
    )


PROFILE_MANAGEMENT_PROFILE = build_profile_management_profile()
