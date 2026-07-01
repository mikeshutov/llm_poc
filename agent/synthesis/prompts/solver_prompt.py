from agent.agentstate.model import AgentState
from agent.prompts.agent_prompt import AgentPrompt, PlanEvidenceStep
from agent.synthesis.prompts.solver_rules import build_solver_rules
from agent.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA


def build_solver_prompt(*, plan_with_evidence: list[PlanEvidenceStep], state: AgentState) -> AgentPrompt:
    return AgentPrompt(
        prompt_kind="synthesis",
        instruction=(
            "Solve the following task or problem. To solve the problem, we have made "
            "step-by-step Plan and retrieved corresponding Evidence to each Plan. "
            "Use them with caution since long evidence might contain irrelevant information."
        ),
        conversation_context=state.conversation_context,
        rules=build_solver_rules(state.request_analysis),
        plan_with_evidence=plan_with_evidence,
        schema=SYNTHESIS_SCHEMA,
        task=state.task,
    )
