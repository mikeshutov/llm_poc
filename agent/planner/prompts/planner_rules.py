def build_planner_rules() -> str:
    rules = [
        "Do not invent tool names. Use tool names exactly as provided.",
        "If you use Brave/WebSearch tools, use at most ONE of them in the entire plan.",
        "Keep each Plan explanation to one sentence.",
        "When multiple tools can be used to gether data use them.",
        "Evidence references must be defined before use. Do not reference evidence unless you have already produced it earlier in the plan.",
        "If final_answer is not null, steps MUST be an empty list.",
        "When attempting to call tools with fields try to fetch their values first from related tools.",
        "For product searches utilize internal tools first before web searches.",
        "Carry forward prior plans, evidence, and constraints that remain relevant."
    ]
    numbered_rules = "\n".join(
        f"{index}) {rule}"
        for index, rule in enumerate(rules, start=1)
    )
    return f"Rules:\n{numbered_rules}"
