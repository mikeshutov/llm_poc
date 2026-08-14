from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable

from request_orchestrator.agents.main_agent.agent import run_agent as run_main_agent
from request_orchestrator.agents.profile_management.agent import run_agent as run_profile_management_agent
from request_orchestrator.constants import REQUEST_ANALYSIS_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from request_orchestrator.shared.profile import load_user_profile
from request_orchestrator.shared.request_analysis.analyze_request import analyze_request
from request_orchestrator.shared.synthesis.synthesis import run_synthesis

PROFILE_MANAGEMENT_EDGE = "profile_management"
MAIN_AGENT_EDGE = "main_agent"
COLLECT_EDGE = "collect"
FANOUT_EDGE = "fanout"
PROFILE_LOADING_EDGE = "load_user_profile"


def _fanout_node(state: MainState) -> MainState:
    state.fan_out_shared_state()
    return MainState(
        task=state.task,
        max_turns=state.max_turns,
        conversation_context=state.conversation_context,
        user_profile=state.user_profile,
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        request_analysis=state.request_analysis,
        agent_states=list(state.agent_states),
        agent_log=state.agent_log,
        result=state.result,
        llm=state.llm,
        conversation_model_config=state.conversation_model_config,
    )


def _run_profile_management_update(state: MainState) -> MainState:
    agent_state = state.get_agent_state("profile_management").clone_for_parallel()
    updated_state = run_profile_management_agent(agent_state)
    state.set_agent_state(updated_state)
    return MainState(
        task=state.task,
        max_turns=state.max_turns,
        conversation_context=state.conversation_context,
        user_profile=state.user_profile,
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        request_analysis=state.request_analysis,
        agent_states=list(state.agent_states),
        agent_log=state.agent_log,
        result=state.result,
        llm=state.llm,
        conversation_model_config=state.conversation_model_config,
    )


def _run_main_agent_update(state: MainState) -> MainState:
    agent_state = state.get_agent_state("main_agent").clone_for_parallel()
    updated_state = run_main_agent(agent_state)
    state.set_agent_state(updated_state)
    return MainState(
        task=state.task,
        max_turns=state.max_turns,
        conversation_context=state.conversation_context,
        user_profile=state.user_profile,
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        request_analysis=state.request_analysis,
        agent_states=list(state.agent_states),
        agent_log=state.agent_log,
        result=state.result,
        llm=state.llm,
        conversation_model_config=state.conversation_model_config,
    )


def _collect_node(state: MainState) -> MainState:
    state.collect_agent_outputs()
    return MainState(
        task=state.task,
        max_turns=state.max_turns,
        conversation_context=state.conversation_context,
        user_profile=state.user_profile,
        conversation_id=state.conversation_id,
        roundtrip_id=state.roundtrip_id,
        request_analysis=state.request_analysis,
        agent_states=list(state.agent_states),
        agent_log=state.agent_log,
        result=state.result,
        llm=state.llm,
        conversation_model_config=state.conversation_model_config,
    )


def _compile_graph():
    builder = StateGraph(MainState)
    builder.add_node(REQUEST_ANALYSIS_EDGE, analyze_request)
    builder.add_node(PROFILE_LOADING_EDGE, load_user_profile)
    builder.add_node(FANOUT_EDGE, _fanout_node)
    builder.add_node(PROFILE_MANAGEMENT_EDGE, _run_profile_management_update)
    builder.add_node(MAIN_AGENT_EDGE, _run_main_agent_update)
    builder.add_node(COLLECT_EDGE, _collect_node)
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(REQUEST_ANALYSIS_EDGE)

    builder.add_edge(REQUEST_ANALYSIS_EDGE, PROFILE_LOADING_EDGE)
    builder.add_edge(PROFILE_LOADING_EDGE, FANOUT_EDGE)
    builder.add_edge(FANOUT_EDGE, PROFILE_MANAGEMENT_EDGE)
    builder.add_edge(PROFILE_MANAGEMENT_EDGE, MAIN_AGENT_EDGE)
    builder.add_edge(MAIN_AGENT_EDGE, COLLECT_EDGE)
    builder.add_edge(COLLECT_EDGE, SYNTHESIZE_EDGE)
    builder.add_edge(SYNTHESIZE_EDGE, END)
    return builder.compile()


@traceable(name="request_orchestrator")
def run_agent(main_state: MainState) -> AgentResult:
    graph = _compile_graph()
    final_state = graph.invoke(
        main_state,
        config={"configurable": {"thread_id": main_state.conversation_id or ""}},
    )
    final = final_state if isinstance(final_state, MainState) else MainState(**final_state)
    return AgentResult(
        answer=list(final.result.answer),
        answer_blocks=[
            SynthesisResultBlock(
                content=block.content,
                evidence_ids=list(block.evidence_ids),
            )
            for block in final.result.answer_blocks
        ],
        next_question=final.result.next_question,
        roundtrip_summary=final.result.roundtrip_summary,
        tool_summary=dict(final.result.tool_summary),
        agent_logs=final.build_agent_logs(),
        used_evidence_ids=list(final.result.used_evidence_ids),
        hydrated_evidence_by_id=dict(final.result.hydrated_evidence_by_id),
    )
