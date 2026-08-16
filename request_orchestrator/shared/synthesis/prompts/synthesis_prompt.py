from conversation.models.conversation_models import ConversationContext
from request_orchestrator.agent_runner.models.agent_profile import DEFAULT_SYNTHESIS_INSTRUCTION
from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep, PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.synthesis.prompts.synthesis_rules import build_synthesis_rules
from request_orchestrator.shared.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA


def _build_synthesis_context(state: AgentState | MainState) -> ConversationContext:
    conversation_context = state.execution_context.conversation_context
    return ConversationContext(
        conversation_summary=conversation_context.conversation_summary,
        latest_conversation_summary=conversation_context.latest_conversation_summary,
        tool_summary=conversation_context.tool_summary,
    )


def build_synthesis_prompt(evidence: list[EvidenceStep], state: AgentState | MainState) -> AgentPrompt:
    tool_categories = (
        [
            category
            for goal in state.request_analysis.goals
            for category in goal.tool_categories
        ]
        if isinstance(state, MainState)
        else list(state.inputs.tool_category_names)
    )
    prompt = AgentPrompt(
        instruction=(
            DEFAULT_SYNTHESIS_INSTRUCTION
            if isinstance(state, MainState)
            else state.agent_profile.synthesis_instruction
        ),
        conversation_context=_build_synthesis_context(state),
        user_profile=state.execution_context.user_profile,
        rules=build_synthesis_rules(tool_categories),
        evidence=evidence,
        schema=SYNTHESIS_SCHEMA,
        task=state.task if isinstance(state, MainState) else state.inputs.task,
    )
    prompt.include_section(
        PromptSectionKeys.USER_PROFILE,
        metadata={"include_tone": True},
    )
    prompt.include_section(PromptSectionKeys.RULES)
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)
    prompt.include_section(PromptSectionKeys.EVIDENCE)
    prompt.include_section(PromptSectionKeys.LATEST_USER_PROMPT)
    prompt.include_section(PromptSectionKeys.SCHEMA)
    return prompt
