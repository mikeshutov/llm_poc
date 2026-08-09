from __future__ import annotations

from langsmith import traceable

from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, REQUEST_ANALYSIS_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from llm.usage import record_llm_call, serialize_llm_call_record
from request_orchestrator.models.agent_state import AgentState, RequestAnalysis
from request_orchestrator.constants import MAIN_AGENT_NAME, REQUEST_ANALYSIS_PROMPT_STEP
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt
from request_orchestrator.agents.main_agent.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt
from rendering.debug import REQUEST_ANALYSIS_KIND


@traceable(name="Request Analysis Node")
def analyze_request(agent_state: AgentState) -> AgentState:
    prompt = build_request_analysis_prompt(agent_state)
    prompt_text = render_agent_prompt(prompt)
    llm = agent_state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
    )
    response = llm.invoke(prompt_text)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=agent_state.resolve_model_for_stage(agent=MAIN_AGENT_MODEL_SCOPE, stage=REQUEST_ANALYSIS_STAGE),
        conversation_id=agent_state.conversation_id,
        roundtrip_id=agent_state.roundtrip_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
        callsite="request_analysis.analyze_request",
    )
    raw = response.content.strip()

    parsed_successfully = True
    try:
        agent_state.request_analysis = RequestAnalysis.model_validate_json(raw)
    except Exception:
        agent_state.request_analysis = RequestAnalysis()
        parsed_successfully = False

    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=REQUEST_ANALYSIS_KIND,
        data={
            "applicable_tool_categories": agent_state.request_analysis.applicable_tool_categories,
            "requested_user_attribute_types": agent_state.request_analysis.requested_user_attribute_types,
            "context_answer_confidence": agent_state.request_analysis.context_answer_confidence,
            "goal": agent_state.request_analysis.goal,
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )
    if parsed_successfully and agent_state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            agent_state.roundtrip_id,
            agent=MAIN_AGENT_NAME,
            prompt_step=REQUEST_ANALYSIS_PROMPT_STEP,
            prompt=prompt_text,
        )

    return agent_state
