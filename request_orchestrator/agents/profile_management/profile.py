from __future__ import annotations

from conversation.models.conversation_model_config import DEFAULT_PROFILE_AGENT_PLANNER_MODEL, PROFILE_AGENT_MODEL_SCOPE
from personalization.profile.models import UserProfile
from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_CATEGORIES, ATTRIBUTE_QUALIFIERS
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile, PROFILE_MANAGEMENT_AGENT_NAME
from request_orchestrator.shared.tool_adapter.profile.set_user_display_name import set_user_display_name
from request_orchestrator.shared.tool_adapter.profile.set_user_first_name import set_user_first_name
from request_orchestrator.shared.tool_adapter.profile.set_user_last_name import set_user_last_name
from request_orchestrator.shared.tool_adapter.profile.update_user_tone import update_user_tone


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
        allowed_categories={'user_attributes'},
        extra_tools=extra_tools,
        default_stage_models={
            "planner": DEFAULT_PROFILE_AGENT_PLANNER_MODEL,
        },
        request_analysis_selectable=True,
        max_turns=5,
        request_analysis_goal=(
            "Review this turn for durable user profile field and user attribute maintenance needs. "
            "If profile work is needed, plan the minimal retrieval and/or update step combination required."
        ),
        planner_instruction=(
            'Maintain durable user profile fields and durable user attributes. '
            'Do not answer the user. '
            'If no profile mutation is needed, return no steps.'
        ),
        planner_rules=(
            'Profile field policy:\n'
            '- `display_name` is the user\'s nickname or preferred short way of being addressed.\n'
            '- Use `set_user_display_name` when the user gives a nickname, preferred short name, or phrasing like `call me ...`.\n'
            '- Only use `set_user_first_name` or `set_user_last_name` when that field is currently missing and the user explicitly provides that real name field in this turn.\n'
            '- Do not store a nickname inside first_name or last_name.\n'
            '- Do not overwrite an existing first_name or last_name with a new guess, paraphrase, or nickname.\n\n'
            'Tone policy:\n'
            '- `tone` stores durable response-style preferences such as concision, formality, directness, humor, and technical depth.\n'
            '- Use `update_user_tone` only when the user clearly expresses a stable preference about how responses should sound or be structured.\n'
            '- Pass a confidence score from 0 to 1. Updates below 0.9 will be rejected by the tool, so only call it when confidence is high.\n'
            '- Update only the tone fields explicitly supported by the user\'s request, and preserve existing tone fields when the user did not revise them.\n'
            '- Do not call `update_user_tone` when the requested tone values already match the stored tone profile.\n'
            '- Do not infer a tone preference from a single task unless the user frames it as an ongoing preference.\n\n'
            'Profile policy:\n'
            '- Store durable user-specific preferences, interests, skills and goals.\n'
            '- Search existing profile data before creating when overlap is plausible.\n'
            '- Prefer merge or update over duplicate creation.\n'
            '- When multiple attributes of the same type already exist, prefer consolidating into the smallest number of coherent entries instead of creating another one.\n'
            '- For categories like `food.likes`, append compatible values to an existing coherent group before creating a new attribute.\n'
            '- Use group_key to preserve meaningful semantic splits, and keep the number of groups as small as possible while staying clear.\n'
            '- When reorganizing, prefer updating or consolidating existing attributes over creating net-new ones unless a new grouped entry is clearly the cleanest shape.\n'
            '- Do not update merely for equivalent wording.\n'
            '- Store concrete values rather than inferred umbrella labels.\n'
            '- Prefer multiple coherent groups over one large mixed attribute.\n\n'
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
