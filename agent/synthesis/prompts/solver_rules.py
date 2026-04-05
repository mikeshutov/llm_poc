from agent.agentstate.model import ClassificationResults
from common.parsing import format_prompt_bullet_list
from tool.tools import TOOL_CATEGORIES

BASE_RULES = [
    "Do not include raw UUIDs or internal identifiers in your response.",
    "When evidence contains a file_path for an image (jpg, jpeg, png, webp), render it using markdown image syntax: ![file_name](/app/static/files/file_name).",
    "When evidence contains a file_path for a document (pdf, txt, docx), render it as a markdown link: [file_name](/app/static/files/file_name).",
]


def build_solver_rules(classification: ClassificationResults | None = None) -> str:
    result_rules = [
        rule
        for cat in (classification.applicable_tool_categories if classification else [])
        if cat in TOOL_CATEGORIES
        for rule in TOOL_CATEGORIES[cat].result_rules
    ]
    rules = BASE_RULES + result_rules
    return format_prompt_bullet_list(rules)
