from __future__ import annotations

from time import perf_counter

from langsmith import traceable
from common.data import repair_common_json_issues, strip_code_fences
from common.logging import create_conversation_event, log_roundtrip_prompt
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, REQUEST_ANALYSIS_STAGE
from llm.usage import record_llm_call, serialize_llm_call_record
from rendering.debug import REQUEST_ANALYSIS_KIND
from request_orchestrator.constants import REQUEST_ANALYSIS_PROMPT_KIND
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.request_analysis import RequestAnalysis
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name
from request_orchestrator.shared.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt

ORCHESTRATOR_AGENT_NAME = "request_orchestrator"


@traceable(name="Request Analysis Node")
def analyze_request(main_state: MainState) -> MainState:
    prompt = build_request_analysis_prompt(main_state)
    prompt_text = prompt.build()
    prompt_input_object = prompt.to_log_input_object()
    execution_context = main_state.execution_context
    model_name = resolve_stage_model_name(
        execution_context=execution_context,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
    )
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=main_state.llm,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=model_name,
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        user_id=execution_context.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
        callsite="request_analysis.analyze_request",
        latency_ms=latency_ms,
        owner_agent_name=ORCHESTRATOR_AGENT_NAME,
        input_object=prompt_input_object,
        output_object={
            "raw_content": response.content,
        },
    )
    raw = repair_common_json_issues(strip_code_fences(response.content.strip()))

    parsed_successfully = True
    try:
        main_state.request_analysis = RequestAnalysis.model_validate_json(raw)
    except Exception:
        main_state.request_analysis = RequestAnalysis()
        parsed_successfully = False

    create_conversation_event(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=REQUEST_ANALYSIS_KIND,
        source=ORCHESTRATOR_AGENT_NAME,
        agent_name=ORCHESTRATOR_AGENT_NAME,
        payload={
            "agent_name": ORCHESTRATOR_AGENT_NAME,
            "kind": REQUEST_ANALYSIS_KIND,
            "data": {
                "goals": [goal.model_dump() for goal in main_state.request_analysis.goals],
                "requested_user_attribute_types": main_state.request_analysis.requested_user_attribute_types,
                "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
            },
        },
    )
    if parsed_successfully and execution_context.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=execution_context.roundtrip_id,
            agent=ORCHESTRATOR_AGENT_NAME,
            prompt_step=REQUEST_ANALYSIS_PROMPT_KIND,
            prompt=prompt_text,
        )

    return main_state
