def build_planner_rules(extra_rules: dict[str, list[str]] | None = None) -> str:
    rules = [
        "Do not invent tool names. Use tool names exactly as provided.",
        "Keep each Plan explanation to one sentence.",
        "Evidence references must be defined before use. Do not reference evidence unless you have already produced it earlier in the plan.",
        "If final_answer is not null, steps MUST be an empty list.",
        "Carry forward prior plans, evidence, and constraints that remain relevant.",
        "If previous iterations have already gathered sufficient data to answer the task, set final_answer and return an empty steps list. Do not repeat tool calls that have already been executed.",
        "When a question depends on the result of another tool try to sequence them when it is obvious from the request and available context.",
        "If a plan could have multiple independent tool calls including calls to the same tool with different inputs, include them all as separate steps in a single plan rather than one at a time.",
    ]
    bullet_rules = "\n".join(f"- {rule}" for rule in rules)
    result = f"Rules:\n{bullet_rules}"
    if extra_rules:
        for tool_name, tool_rules in extra_rules.items():
            bullet_tool_rules = "\n".join(f"- {rule}" for rule in tool_rules)
            result += f"\n\n{tool_name} Rules:\n{bullet_tool_rules}"
    return result
