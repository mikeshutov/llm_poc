from __future__ import annotations

from time import perf_counter

from langsmith import traceable

from common.data import repair_common_json_issues, strip_code_fences
from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, REQUEST_ANALYSIS_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from llm.usage import record_llm_call, serialize_llm_call_record
from rendering.debug import REQUEST_ANALYSIS_KIND
from request_orchestrator.constants import MAIN_AGENT_NAME, REQUEST_ANALYSIS_PROMPT_STEP
from request_orchestrator.models.agent_state import RequestAnalysis
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt

ORCHESTRATOR_AGENT_NAME = "request_orchestrator"


@traceable(name="Request Analysis Node")
def analyze_request(main_state: MainState) -> MainState:
    prompt = build_request_analysis_prompt(main_state)
    prompt_text = prompt.prompt_text()
    prompt_input_object = prompt.to_log_input_object()
    main_agent_state = main_state.get_agent_state(MAIN_AGENT_NAME)
    llm = main_agent_state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=main_agent_state.resolve_model_for_stage(agent=MAIN_AGENT_MODEL_SCOPE, stage=REQUEST_ANALYSIS_STAGE),
        conversation_id=main_state.conversation_id,
        roundtrip_id=main_state.roundtrip_id,
        user_id=main_state.user_profile.user_id,
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
        callsite="request_analysis.analyze_request",
        latency_ms=latency_ms,
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

    main_state.agent_log.add(
        agent_name=ORCHESTRATOR_AGENT_NAME,
        kind=REQUEST_ANALYSIS_KIND,
        data={
            "applicable_tool_categories": main_state.request_analysis.applicable_tool_categories,
            "requested_user_attribute_types": main_state.request_analysis.requested_user_attribute_types,
            "goal": main_state.request_analysis.goal,
            "llm_usage": None if llm_call is None else serialize_llm_call_record(llm_call),
        },
    )
    if parsed_successfully and main_state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            main_state.roundtrip_id,
            agent=MAIN_AGENT_NAME,
            prompt_step=REQUEST_ANALYSIS_PROMPT_STEP,
            prompt=prompt_text,
        )

    return main_state
