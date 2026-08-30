from common.data import format_prompt_bullet_list
from tool.tools import TOOL_CATEGORIES

BASE_RULES = [
    "Do not include raw UUIDs or internal identifiers in your response.",
]
TONE_ADAPTATION_RULE = (
    "Apply the user's tone profile preferences when writing the response. Adapt wording, structure, and level of detail without mentioning the preferences themselves."
)
PROFILE_PERSONALIZATION_RULE = (
    "When relevant, use the user's profile information to personalize the response naturally but don't overdo it sometimes no usage is needed."
)


def build_synthesis_rules(
    tool_categories: list[str] | None = None,
    *,
    apply_tone_preferences: bool = False,
    apply_profile_personalization: bool = False,
) -> str:
    result_rules = [
        rule
        for cat in (tool_categories or [])
        if cat in TOOL_CATEGORIES
        for rule in TOOL_CATEGORIES[cat].result_rules
    ]
    rules = BASE_RULES + result_rules
    if apply_tone_preferences:
        rules.append(TONE_ADAPTATION_RULE)
    if apply_profile_personalization:
        rules.append(PROFILE_PERSONALIZATION_RULE)
    return format_prompt_bullet_list(rules)
