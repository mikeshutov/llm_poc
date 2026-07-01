from __future__ import annotations

from langsmith import traceable

from agent.agentstate.model import AgentState, RequestAnalysis
from agent.prompt_constants import MAIN_AGENT_NAME, REQUEST_ANALYSIS_PROMPT_STEP
from agent.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt
from conversation.repository.repo_factory import get_conversation_repo
from rendering.debug import build_request_analysis_status_message, emit_status_message


@traceable(name="Request Analysis Node")
def analyze_request(agent_state: AgentState) -> AgentState:
    prompt = build_request_analysis_prompt(agent_state)
    raw = agent_state.llm.invoke(prompt.to_string()).content.strip()

    parsed_successfully = True
    try:
        agent_state.request_analysis = RequestAnalysis.model_validate_json(raw)
    except Exception:
        agent_state.request_analysis = RequestAnalysis()
        parsed_successfully = False

    emit_status_message(
        build_request_analysis_status_message(
            agent_state.request_analysis.applicable_tool_categories,
            agent_state.request_analysis.context_answer_confidence,
            agent_state.request_analysis.goal,
        )
    )
    if (
        not agent_state.request_analysis.requires_tools
        and agent_state.request_analysis.context_answer_confidence >= 0.8
    ):
        agent_state.goal_reached = True

    if parsed_successfully and agent_state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            agent_state.roundtrip_id,
            agent=MAIN_AGENT_NAME,
            prompt_step=REQUEST_ANALYSIS_PROMPT_STEP,
            prompt=prompt.to_string(),
        )

    return agent_state
