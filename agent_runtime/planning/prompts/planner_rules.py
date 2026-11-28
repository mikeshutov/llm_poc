from common.data import format_prompt_bullet_list

DEFAULT_PLANNER_PROMPT_INSTRUCTION = "You are a planning agent."

WORKING_MEMORY_LIFECYCLE_RULE = (
    "Use update_memory_state to maintain your current task state and working memory: objective, "
    "progress, findings, decisions, supporting evidence IDs, and remaining work. "
    "At the start of every reasoning cycle, review every tool result available since "
    "the previous working-memory update. Reconcile those results with the current "
    "understanding, record relevant findings, decisions, gaps, and supporting evidence "
    "IDs in working memory, and update working memory before choosing the next action "
    "batch. Do not carry unprocessed tool results into another reasoning cycle. "
    "Working-memory updates may be planned and executed alongside other independent "
    "tool calls. Update working memory again after the final available tool-result batch "
    "before finishing. A successful working-memory update retains the selected state "
    "and automatically condenses previously viewed eligible tool exchanges from the current roundtrip. "
    "State-changing tools run sequentially before concurrent-safe read tools; new batch results remain "
    "available for the next reasoning cycle. "
    "When the task is complete, make one final successful update_memory_state with "
    "task_completed=true as the only tool call in its batch before returning the final response; do not rely on an empty "
    "action set to indicate that state was saved."
)

BASE_PLANNER_RULES = [
    "Use the available tools to complete the task.",
    "Return tool steps only when they meaningfully advance the goal.",
    "Do not repeat materially equivalent tool calls that have already been executed.",
    "You may use already-available tool results from previous work when deciding what to do next.",
    "Prefer the smallest useful set of tool calls.",
    "Each tool call must include an affirmative, factual, testable hypothesis that can be verified against the returned evidence, not an action or justification.",
    "Set evaluation_fields to the minimum data necessary to validate the hypothesis. Do not include the record name or the full result schema by default.",
]

REACT_DECISION_RULES = [
    "Inspect the observed tool results before choosing the next action.",
    "Return only actions that materially advance the user's goal.",
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
