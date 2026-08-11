from conversation.models.conversation_models import ConversationContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import SYNTHESIS_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PlanEvidenceStep
from request_orchestrator.shared.synthesis.prompts.solver_rules import build_solver_rules
from request_orchestrator.shared.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA

def _build_synthesis_context(state: AgentState) -> ConversationContext:
    return ConversationContext(
        latest_conversation_summary=state.conversation_context.latest_conversation_summary,
    )


def build_solver_prompt(plan_with_evidence: list[PlanEvidenceStep], state: AgentState) -> AgentPrompt:
    return AgentPrompt(
        prompt_kind=SYNTHESIS_PROMPT_KIND,
        instruction=state.agent_profile.synthesis_instruction,
        conversation_context=_build_synthesis_context(state),
        user_profile=state.user_profile,
        rules=build_solver_rules(state.request_analysis),
        plan_with_evidence=plan_with_evidence,
        schema=SYNTHESIS_SCHEMA,
        task=state.task,
    )
