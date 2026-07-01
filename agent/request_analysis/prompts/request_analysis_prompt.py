from agent.agentstate.model import AgentState, UserProfile
from agent.request_analysis.prompts.request_analysis_schema_prompt import REQUEST_ANALYSIS_SCHEMA
from agent.prompts.agent_prompt import AgentPrompt
from tool.tools import TOOL_CATEGORIES


def build_request_analysis_prompt(agent_state: AgentState) -> AgentPrompt:
    category_lines = "\n".join(
        f"- Category: {name} | Category Description: {cat.description}"
        for name, cat in TOOL_CATEGORIES.items()
    )

    return AgentPrompt(
        prompt_kind="request_analysis",
        instruction=(
            "You are a request analyzer. "
            "Infer the user's goal, decide whether tools are required, and return the relevant category names. "
            "When conversation context includes recent_roundtrip_tool_summaries, treat them as the highest-priority tool-use context. "
            "Use the older string tool_summary only as fallback context when the structured roundtrip tool summaries are absent or incomplete."
        ),
        conversation_context=agent_state.conversation_context,
        user_profile=UserProfile(geometadata=agent_state.geometadata),
        available_tool_categories=category_lines,
        schema=REQUEST_ANALYSIS_SCHEMA,
        task=agent_state.task,
    )
