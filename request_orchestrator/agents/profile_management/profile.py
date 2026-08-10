from __future__ import annotations

from personalization.profile.models import UserProfile
from request_orchestrator.agents.models.agent_profile import AgentProfile
from request_orchestrator.shared.tool_adapter.profile.set_user_display_name import set_user_display_name
from request_orchestrator.shared.tool_adapter.profile.set_user_first_name import set_user_first_name
from request_orchestrator.shared.tool_adapter.profile.set_user_last_name import set_user_last_name


def build_profile_management_profile(user_profile: UserProfile | None = None) -> AgentProfile:
    extra_tools = [set_user_display_name]

    if user_profile is None or not (user_profile.first_name or '').strip():
        extra_tools.append(set_user_first_name)
    if user_profile is None or not (user_profile.last_name or '').strip():
        extra_tools.append(set_user_last_name)

    return AgentProfile(
        name='profile_management',
        allowed_categories={'user_attributes'},
        extra_tools=extra_tools,
        planner_instruction=(
            'Maintain durable user profile fields and durable user attributes. '
            'Do not answer the user. '
            'If no profile mutation is needed, return no steps.'
        ),
        persist_tool_calls=False,
        planner_rules=(
            'Profile field policy:\n'
            '- `display_name` is the user\'s nickname or preferred short way of being addressed.\n'
            '- Use `set_user_display_name` when the user gives a nickname, preferred short name, or phrasing like `call me ...`.\n'
            '- Only use `set_user_first_name` or `set_user_last_name` when that field is currently missing and the user explicitly provides that real name field in this turn.\n'
            '- Do not store a nickname inside first_name or last_name.\n'
            '- Do not overwrite an existing first_name or last_name with a new guess, paraphrase, or nickname.\n\n'
            'Profile policy:\n'
            '- Store durable user-specific preferences, interests, skills and goals.\n'
            '- Search existing profile data before creating when overlap is plausible.\n'
            '- Prefer merge or update over duplicate creation.\n'
            '- When multiple attributes of the same type already exist, prefer consolidating into the smallest number of coherent entries instead of creating another one.\n'
            '- For categories like `food.likes`, append compatible values to an existing coherent group before creating a new attribute.\n'
            '- Use group_key when it helps preserve meaningful semantic splits, and keep the number of groups as small as possible while staying clear.\n'
            '- Temporary cleanup rule: you may reorganize existing attributes when the current structure is duplicative, fragmented, overly mixed, or poorly grouped, even if the user did not provide a brand-new fact in this turn.\n'
            '- When reorganizing, prefer updating or consolidating existing attributes over creating net-new ones unless a new grouped entry is clearly the cleanest shape.\n'
            '- Do not update merely for equivalent wording.\n'
            '- Store concrete values rather than inferred umbrella labels.\n'
            '- Prefer multiple coherent groups over one large mixed attribute.\n\n'
            'Execution rules:\n'
            '- Do not repeat executed calls.\n'
            '- Reuse previous evidence.\n'
            '- Keep plans to one sentence.'
        ),
    )


PROFILE_MANAGEMENT_PROFILE = build_profile_management_profile()
