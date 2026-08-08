from __future__ import annotations

from request_orchestrator.agents.models.agent_profile import AgentProfile


PROFILE_MANAGEMENT_PROFILE = AgentProfile(
    name='profile_management',
    allowed_categories={'user_attributes'},
    extra_tools=[],
    planner_instruction=(
        'Maintain durable user profile attributes. '
        'Do not answer the user. '
        'If no profile mutation is needed, return no steps.'
    ),
    persist_tool_calls=False,
    planner_rules=(
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
