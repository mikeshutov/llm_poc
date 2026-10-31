from __future__ import annotations

from llm.conversation_model_config import PROFILE_AGENT_MODEL_SCOPE
from personalization.profile.models import UserProfile
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_CATEGORIES, ATTRIBUTE_QUALIFIERS
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile, PROFILE_MANAGEMENT_AGENT_NAME
from request_orchestrator.shared.tool_adapter.profile import (
    set_user_display_name,
    set_user_first_name,
    set_user_last_name,
    update_user_tone,
)


def build_profile_management_profile(user_profile: UserProfile | None = None) -> AgentProfile:
    extra_tools = [set_user_display_name, update_user_tone]
    attribute_prefixes = ", ".join(ATTRIBUTE_CATEGORIES)
    attribute_suffixes = ", ".join(ATTRIBUTE_QUALIFIERS)

    if user_profile is None or not (user_profile.first_name or '').strip():
        extra_tools.append(set_user_first_name)
    if user_profile is None or not (user_profile.last_name or '').strip():
        extra_tools.append(set_user_last_name)

    return AgentProfile(
        name=PROFILE_MANAGEMENT_AGENT_NAME,
        scope=PROFILE_AGENT_MODEL_SCOPE,
        description=(
            "Maintain durable user profile fields and attributes. Use this whenever "
            "the user states or hints at likes, dislikes, preferences, or other durable "
            "personal details that should update their user attributes."
        ),
        allowed_categories={'user_attributes'},
        extra_tools=extra_tools,
        delegatable=True,
        max_turns=5,
        planner_instruction=(
            'Maintain durable user profile fields and durable user attributes. '
            'Do not answer the user. '
            'If no profile mutation is needed, return no steps.'
        ),
        planner_rules=(
            'Attribute Type Rules:\n'
            f'- Available attribute prefixes: {attribute_prefixes}.\n'
            f'- Available attribute suffixes: {attribute_suffixes}.\n'
            '- Requested or updated attribute types must use the format prefix.suffix such as food.likes or projects.goals.\n\n'
            'Execution rules:\n'
            '- Do not repeat executed calls.\n'
            '- Do not make one planned tool step depend on another step\'s output. Each step must stand on its own inputs for now.\n'
            '- Keep plans to one sentence.'
        ),
    )


PROFILE_MANAGEMENT_PROFILE = build_profile_management_profile()
