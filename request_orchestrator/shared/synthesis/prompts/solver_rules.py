from request_orchestrator.models.agent_state import RequestAnalysis
from common.data import format_prompt_bullet_list
from tool.tools import TOOL_CATEGORIES

BASE_RULES = [
    "Do not include raw UUIDs or internal identifiers in your response.",
]


def build_solver_rules(request_analysis: RequestAnalysis | None = None) -> str:
    result_rules = [
        rule
        for cat in (request_analysis.applicable_tool_categories if request_analysis else [])
        if cat in TOOL_CATEGORIES
        for rule in TOOL_CATEGORIES[cat].result_rules
    ]
    rules = BASE_RULES + result_rules
    return format_prompt_bullet_list(rules)

