from common.data import format_prompt_bullet_list


BASE_PLANNER_RULES = [
    "Do not invent tool names. Use tool names exactly as provided.",
    "Return tool steps only when they meaningfully advance the goal.",
    "Do not repeat materially equivalent tool calls that have already been executed.",
    "Keep each Plan explanation to one sentence.",
    "Do not make one planned tool step depend on another step's output. For now, every step must be runnable from its own arguments alone.",
    "You may use already-available tool results from previous work when deciding what to do next.",
    "Prefer the smallest useful set of tool calls.",
    "If there are no tools to use to fulfill the request, return no tool calls.",
    "Tool Call Hypotheses:",
    "For each tool call, generate a hypothesis describing what the tool result is expected to establish.",
    "Express the hypothesis as a factual, testable statement, not an action, instruction, goal, or justification.",
    "Describe the expected evidence, not why the tool is being called.",
    "The hypothesis must be independently verifiable using the tool's returned data.",
    "Use affirmative, declarative language. Avoid phrases such as 'Find', 'Search', 'Retrieve', 'Determine', or 'Help answer'.",
    "Do not assert information that cannot reasonably be established by the tool's output.",
    "Set evaluation_fields to the minimum data necessary to validate the hypothesis. Do not include the record name or the full result schema by default.",
    "Bad hypothesis: 'Find available beef recipes to answer the user\'s request.'",
    "Good hypothesis: 'The search results contain multiple recipes categorized as beef.'",
    "Bad hypothesis: 'Check whether the product is available.'",
    "Good hypothesis: 'The product search results indicate that the product is available for purchase.'",
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
