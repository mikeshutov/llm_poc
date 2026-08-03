from agent.agentstate.model import AgentState
from agent.prompt_constants import REQUEST_ANALYSIS_PROMPT_KIND
from agent.request_analysis.prompts.request_analysis_schema_prompt import REQUEST_ANALYSIS_SCHEMA
from agent.prompts.agent_prompt import AgentPrompt
from tool.tools import TOOL_CATEGORIES


def build_request_analysis_prompt(agent_state: AgentState) -> AgentPrompt:
    category_lines = "\n".join(
        f"- Category: {name} | Category Description: {cat.description}"
        for name, cat in TOOL_CATEGORIES.items()
    )

    return AgentPrompt(
        prompt_kind=REQUEST_ANALYSIS_PROMPT_KIND,
        instruction=(
            "You are a request analyzer. "
            "Infer the user's goal, decide whether tools are required, and return the relevant category names. "
            "Use recent_roundtrip_tool_summaries as helpful context about prior tool usage, entities, produced fields, and freshness. "
            "Use recent_roundtrips when the user refers to something said in a recent prior message or to a recent turn summary. "
            "Use the older string tool_summary only as fallback context when the structured roundtrip tool summaries are absent or incomplete. "
            "If the user is asking about something previously discussed, suggested, decided, or mentioned, include the memories category. "
            "The prompt already includes user_profile, including stored user attributes. If that profile context appears sufficient, tools may be unnecessary; include the user_attributes category when the task requires creating, updating, deactivating, or more precisely retrieving durable user-specific profile data. "
            "If there is meaningful doubt that the available context alone is sufficient, set requires_tools to true."
        ),
        conversation_context=agent_state.conversation_context,
        user_profile=agent_state.user_profile,
        available_tool_categories=category_lines,
        schema=REQUEST_ANALYSIS_SCHEMA,
        task=agent_state.task,
    )
