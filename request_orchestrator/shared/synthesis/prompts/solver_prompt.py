from conversation.models.conversation_models import ConversationContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import SYNTHESIS_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep
from request_orchestrator.shared.synthesis.prompts.solver_rules import build_solver_rules
from request_orchestrator.shared.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA

def _build_synthesis_context(state: AgentState) -> ConversationContext:
    return ConversationContext(
        conversation_summary=state.conversation_context.conversation_summary,
        latest_conversation_summary=state.conversation_context.latest_conversation_summary,
        tool_summary=state.conversation_context.tool_summary,
    )


def build_solver_prompt(evidence: list[EvidenceStep], state: AgentState) -> AgentPrompt:
    prompt = AgentPrompt(
        prompt_kind=SYNTHESIS_PROMPT_KIND,
        instruction=state.agent_profile.synthesis_instruction,
        conversation_context=_build_synthesis_context(state),
        user_profile=state.user_profile,
        rules=build_solver_rules(state.request_analysis),
        evidence=evidence,
        schema=SYNTHESIS_SCHEMA,
        task=state.task,
    )
    prompt.include_user_profile(include_tone=True)
    prompt.include_rules_section()
    prompt.include_conversation_context()
    prompt.include_evidence()
    prompt.include_text("Now solve the question or task using the evidence above.")
    prompt.include_latest_user_prompt()
    prompt.include_schema_raw()
    return prompt
