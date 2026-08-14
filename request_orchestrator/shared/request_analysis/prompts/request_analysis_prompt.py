from personalization.user_attributes.models.user_attribute_types import ATTRIBUTE_CATEGORIES, ATTRIBUTE_QUALIFIERS
from request_orchestrator.constants import REQUEST_ANALYSIS_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.request_analysis.prompts.request_analysis_schema_prompt import REQUEST_ANALYSIS_SCHEMA


def build_request_analysis_prompt(main_state: MainState) -> AgentPrompt:
    available_agents = main_state.available_request_analysis_agents_payload()
    attribute_prefixes = ", ".join(ATTRIBUTE_CATEGORIES)
    attribute_suffixes = ", ".join(ATTRIBUTE_QUALIFIERS)

    prompt = AgentPrompt(
        prompt_kind=REQUEST_ANALYSIS_PROMPT_KIND,
        instruction=(
            "You are a request analyzer. "
            "Infer one or more self-contained goals for the available agents. "
            "Each goal must name the target agent, the concrete goal for that agent, and the tool categories that agent should be allowed to use. "
            "Only choose from the provided available agents. "
            "Make every goal self-contained for downstream steps because the full conversation context will not be passed through later. "
            "Each goal should capture the actual objective plus any relevant conversation-derived constraints, references, or continuity needed for planning and synthesis. "
            "Name the concrete topic, subject, entity, or item in each goal instead of using vague placeholders like topic, subject, it, them, or the above. "
            "For lookup or search requests, explicitly state what should be searched for so downstream steps do not need the original conversation to know the target. "
            "Use recent_roundtrip_tool_summaries as helpful context about prior tool usage, entities, produced fields, and freshness. "
            "Use recent_roundtrips when the user refers to something said in a recent prior message or to a recent turn summary. "
            "If the user is asking about something previously discussed, suggested, decided, or mentioned, include the memories category in the relevant agent goal. "
            "The prompt includes geometadata but does not pre-load stored user attributes for this step. "
            "When stored user attributes would be beneficial for the request, set requested_user_attribute_types to an array of specific attribute types to load using the available attribute prefixes and suffixes. "
            f"Available attribute prefixes: {attribute_prefixes}. "
            f"Available attribute suffixes: {attribute_suffixes}. "
            "Requested attribute types must use the format prefix.suffix such as food.likes or projects.goals. "
            "Only request user attribute types that would materially help with the current request. "
        ),
        conversation_context=main_state.conversation_context,
        user_profile=main_state.user_profile,
        available_tool_categories=AgentPrompt._serialize_json(available_agents),
        schema=REQUEST_ANALYSIS_SCHEMA,
        task=main_state.task,
    )
    prompt.include_user_profile()
    prompt.include_conversation_context(heading="Conversation context (JSON):")
    prompt.include_available_tool_categories(heading="Available agents (JSON):")
    prompt.include_latest_user_prompt()
    prompt.include_schema_as_response_label()
    return prompt
