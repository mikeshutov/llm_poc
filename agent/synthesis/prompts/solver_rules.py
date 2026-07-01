from agent.agentstate.model import RequestAnalysis
from common.parsing import format_prompt_bullet_list
from tool.tools import TOOL_CATEGORIES

BASE_RULES = [
    "Do not include raw UUIDs or internal identifiers in your response.",
    "When evidence contains a file_path (e.g. 'static/files/foo.pdf'), render it as a clickable link or image by prepending '/app/' to form the URL ('/app/static/files/foo.pdf'). Use image syntax ![file_name](url) for images (jpg, jpeg, png, webp) and link syntax [file_name](url) for all other files. Convert spaces in file names to %20.",
    "Use recent_roundtrip_tool_summaries as the highest-priority summary of recent tool usage, produced fields, entities, and freshness.",
    "Use the older string tool_summary only as fallback context when recent_roundtrip_tool_summaries are absent or incomplete.",
    "When recent_roundtrip_tool_summaries already provide sufficient fresh evidence, avoid implying that new tool calls were needed for the answer.",
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
