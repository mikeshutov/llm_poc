from conversation.models.conversation_models import ConversationContext
from request_orchestrator.constants import SYNTHESIS_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.synthesis.prompts.synthesis_rules import build_synthesis_rules
from request_orchestrator.shared.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA


def _build_synthesis_context(state: AgentState | MainState) -> ConversationContext:
    return ConversationContext(
        conversation_summary=state.conversation_context.conversation_summary,
        latest_conversation_summary=state.conversation_context.latest_conversation_summary,
        tool_summary=state.conversation_context.tool_summary,
    )


def build_synthesis_prompt(evidence: list[EvidenceStep], state: AgentState | MainState) -> AgentPrompt:
    prompt = AgentPrompt(
        prompt_kind=SYNTHESIS_PROMPT_KIND,
        instruction=(
            state.synthesis_instruction
            if isinstance(state, MainState)
            else state.agent_profile.synthesis_instruction
        ),
        conversation_context=_build_synthesis_context(state),
        user_profile=state.user_profile,
        rules=build_synthesis_rules(state.request_analysis),
        evidence=evidence,
        schema=SYNTHESIS_SCHEMA,
        task=state.task,
    )
    prompt.include_user_profile(include_tone=True)
    prompt.include_rules_section()
    prompt.include_conversation_context()
    prompt.include_evidence()
    prompt.include_latest_user_prompt()
    prompt.include_schema_raw()
    return prompt
