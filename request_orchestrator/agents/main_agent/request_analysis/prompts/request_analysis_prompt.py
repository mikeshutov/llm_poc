from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_CATEGORIES, ATTRIBUTE_QUALIFIERS
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import REQUEST_ANALYSIS_PROMPT_KIND
from request_orchestrator.agents.main_agent.request_analysis.prompts.request_analysis_schema_prompt import REQUEST_ANALYSIS_SCHEMA
from request_orchestrator.models.agent_prompt import AgentPrompt


def build_request_analysis_prompt(agent_state: AgentState) -> AgentPrompt:
    category_lines = "\n".join(
        f"- Category: {name} | Category Description: {category.description}"
        for name, category in agent_state.agent_profile.allowed_tool_categories().items()
    )
    attribute_prefixes = ", ".join(ATTRIBUTE_CATEGORIES)
    attribute_suffixes = ", ".join(ATTRIBUTE_QUALIFIERS)

    prompt = AgentPrompt(
        prompt_kind=REQUEST_ANALYSIS_PROMPT_KIND,
        instruction=(
            "You are a request analyzer. "
            "Infer the user's goal, when tools are required return the relevant category names. "
            "Make the goal self-contained for downstream steps because the full conversation context will not be passed through later. "
            "The goal should capture the actual objective plus any relevant conversation-derived constraints, references, or continuity needed for planning and synthesis. "
            "Name the concrete topic, subject, entity, or item in the goal instead of using vague placeholders like topic, subject, it, them, or the above. "
            "For lookup or search requests, explicitly state what should be searched for so downstream steps do not need the original conversation to know the target. "
            "Use recent_roundtrip_tool_summaries as helpful context about prior tool usage, entities, produced fields, and freshness. "
            "Use recent_roundtrips when the user refers to something said in a recent prior message or to a recent turn summary. "
            "If the user is asking about something previously discussed, suggested, decided, or mentioned, include the memories category. "
            "The prompt includes geometadata but does not pre-load stored user attributes for this step. "
            "When stored user attributes would be beneficial for the request, set requested_user_attribute_types to an array of specific attribute types to load using the available attribute prefixes and suffixes. "
            f"Available attribute prefixes: {attribute_prefixes}. "
            f"Available attribute suffixes: {attribute_suffixes}. "
            "Requested attribute types must use the format prefix.suffix such as food.likes or projects.goals. "
            "Only request user attribute types that would materially help with the current request. "
        ),
        conversation_context=agent_state.conversation_context,
        user_profile=agent_state.user_profile,
        available_tool_categories=category_lines,
        schema=REQUEST_ANALYSIS_SCHEMA,
        task=agent_state.task,
    )
    prompt.include_user_profile()
    prompt.include_conversation_context(heading="Conversation context (JSON):")
    prompt.include_available_tool_categories()
    prompt.include_latest_user_prompt()
    prompt.include_schema_as_response_label()
    return prompt
