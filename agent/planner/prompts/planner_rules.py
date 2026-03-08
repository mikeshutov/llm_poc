def build_planner_rules(extra_rules: dict[str, list[str]] | None = None) -> str:
    rules = [
        "Do not invent tool names. Use tool names exactly as provided.",
        "Keep each Plan explanation to one sentence.",
        "When multiple tools can be used to gether data use them.",
        "Evidence references must be defined before use. Do not reference evidence unless you have already produced it earlier in the plan.",
        "If final_answer is not null, steps MUST be an empty list.",
        "When attempting to call tools with fields try to fetch their values first from related tools.",
        "Carry forward prior plans, evidence, and constraints that remain relevant."
        "When a question depends on the result of another tool try to sequence them dependency is obvious from the request and available context."
    ]
    bullet_rules = "\n".join(f"- {rule}" for rule in rules)
    result = f"Rules:\n{bullet_rules}"
    if extra_rules:
        for tool_name, tool_rules in extra_rules.items():
            bullet_tool_rules = "\n".join(f"- {rule}" for rule in tool_rules)
            result += f"\n\n{tool_name} Rules:\n{bullet_tool_rules}"
    return result
