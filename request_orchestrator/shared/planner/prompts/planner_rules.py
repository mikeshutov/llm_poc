from common.parsing import format_prompt_bullet_list


BASE_PLANNER_RULES = [
    "Do not invent tool names. Use tool names exactly as provided.",
    "Return tool steps only when they meaningfully advance the goal.",
    "Do not repeat materially equivalent tool calls that have already been executed.",
    "Keep each Plan explanation to one sentence.",
    "Sequence dependent calls when a later call requires an earlier result. Include independent calls together in the same plan when useful.",
    "Prefer the smallest useful set of tool calls.",
]

MAIN_AGENT_PLANNER_RULES = []

PROFILE_MANAGEMENT_PLANNER_RULES = [
    "Prefer the smallest useful set of retrieval and mutation steps needed to maintain the profile.",
]


def build_planner_rules(
    extra_rules: dict[str, list[str]] | None = None,
    *,
    include_contextual_rules: bool = True,
) -> str:
    rules = list(BASE_PLANNER_RULES)
    if include_contextual_rules:
        rules.extend(MAIN_AGENT_PLANNER_RULES)
    else:
        rules.extend(PROFILE_MANAGEMENT_PLANNER_RULES)

    result = f"Rules:\n{format_prompt_bullet_list(rules)}"
    if extra_rules:
        for tool_name, tool_rules in extra_rules.items():
            result += f"\n{tool_name} Rules:\n{format_prompt_bullet_list(tool_rules)}"
    return result
