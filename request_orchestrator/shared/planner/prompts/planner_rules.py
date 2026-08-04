from common.parsing import format_prompt_bullet_list


def build_planner_rules(
    extra_rules: dict[str, list[str]] | None = None,
    *,
    requires_tools: bool = False,
    has_prior_tool_results: bool = False,
) -> str:
    rules = [
        "Do not invent tool names. Use tool names exactly as provided.",
        "Keep each Plan explanation to one sentence.",
        "Evidence references must be defined before use. Do not reference evidence unless you have already produced it earlier in the plan.",
        "If final_answer is not null, steps MUST be an empty list.",
        "Do not repeat tool calls that have already been executed.",
        "Carry forward prior plans, evidence, and constraints that remain relevant.",
        "Use recent_roundtrip_tool_summaries as helpful context about prior tool usage, entities, produced fields, and freshness.",
        "Use recent_roundtrips when the user refers to a recent prior message, wording from earlier in the conversation, or a recent turn summary.",
        "Use the older string tool_summary only as fallback context when recent_roundtrip_tool_summaries are absent or incomplete.",
        "If previous iterations have already gathered sufficient data to answer the task, set final_answer and return an empty steps list.",
        "If there is meaningful doubt that prior context or previous tool results are sufficient, plan tool calls to verify or fill the gap.",
        "When a question depends on the result of another tool try to sequence them when it is obvious from the request and available context.",
        "If a plan could have multiple independent tool calls including calls to the same tool with different inputs, include them all as separate steps in a single plan rather than one at a time.",
        "Utilize multiple tools when it is appropriate to get full context.",
    ]

    if requires_tools and not has_prior_tool_results:
        rules.append(
            "Request analysis determined that tool use is required. Because no tool results have been gathered yet, you must return at least one tool step and final_answer must be null on this pass."
        )

    result = f"Rules:\n{format_prompt_bullet_list(rules)}"
    if extra_rules:
        for tool_name, tool_rules in extra_rules.items():
            result += f"\n{tool_name} Rules:\n{format_prompt_bullet_list(tool_rules)}"
    return result
