from common.data import format_prompt_bullet_list


BASE_PLANNER_RULES = [
    "Use the available tools to complete the task.",
    "Return tool steps only when they meaningfully advance the goal.",
    "Do not repeat materially equivalent tool calls that have already been executed.",
    "You may use already-available tool results from previous work when deciding what to do next.",
    "Prefer the smallest useful set of tool calls.",
    "If there are no tools to use to fulfill the request, return no tool calls.",
    "Each tool call must include an affirmative, factual, testable hypothesis that can be verified against the returned evidence, not an action or justification.",
    "Set evaluation_fields to the minimum data necessary to validate the hypothesis. Do not include the record name or the full result schema by default.",
]

REACT_DECISION_RULES = [
    "Inspect the observed tool results before choosing the next action.",
    "Return only actions that materially advance the user's goal.",
    "Return no tool calls when the goal is satisfied or no useful action remains.",
    "Do not repeat materially equivalent tool calls that have already been executed.",
]


def build_planner_rules(
    extra_rules: dict[str, list[str]] | None = None,
) -> str:
    rules = list(BASE_PLANNER_RULES)

    result = f"Rules:\n{format_prompt_bullet_list(rules)}"
    if extra_rules:
        for tool_name, tool_rules in extra_rules.items():
            result += f"\n{tool_name} Rules:\n{format_prompt_bullet_list(tool_rules)}"
    return result
