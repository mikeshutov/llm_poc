from __future__ import annotations

from dataclasses import dataclass
import json

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, ToolMessage

from common.data import repair_common_json_issues, strip_code_fences
from common.data import bounded_json_value
from common.logging import record_node_log
from agent_runtime.constants import EXECUTE_TOOLS_EDGE
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.synthesized_result import SynthesisResult, SynthesisResultBlock
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.strategies.react.decision import DecisionStatus, run_decision
from agents.main.prompts import build_reasoner_prompt


def _parse_answer(raw_content: str) -> SynthesisResult:
    raw = repair_common_json_issues(strip_code_fences(raw_content or ""))
    try:
        return SynthesisResult.model_validate_json(raw)
    except Exception:
        text = raw.strip() or "I was unable to produce a response."
        return SynthesisResult(
            result=[SynthesisResultBlock(content=text)],
            roundtrip_summary=text[:1_000],
        )


def run_reasoner(state: AgentState) -> AgentState:
    try:
        decision = run_decision(
            state,
            callsite="main_reasoner.run_reasoner",
            prompt_builder=build_reasoner_prompt,
            message_history=state.reasoner_message_history,
            tool_choice=("required" if state.node_states.planner.plan_count == 0 else None),
        )
    except Exception as exc:
        record_node_log(
            roundtrip_id=state.execution_context.roundtrip_id,
            node_name="reasoner",
            stage="invoke",
            event_type="reasoner_failure",
            level="error",
            message="Reasoner invocation failed",
            payload={"error": str(exc), "error_type": type(exc).__name__},
        )
        state.node_states.planner.plan = None
        state.node_states.planner.plan_count += 1
        state.node_states.planner.decision_status = DecisionStatus.BLOCKED
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
        state.reasoner_result = SynthesisResult(
            result=[SynthesisResultBlock(content="I was unable to complete that request.")],
            roundtrip_summary="The reasoner failed before producing an answer.",
        )
        return state

    state.node_states.planner.plan = decision.plan
    state.node_states.planner.plan_count += 1
    state.node_states.planner.decision_status = decision.status
    if decision.status == DecisionStatus.SATISFIED:
        state.reasoner_result = _parse_answer(decision.raw_content)
        state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS)
    elif decision.status == DecisionStatus.BLOCKED:
        state.reasoner_result = _parse_answer(decision.raw_content)
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
    return state


def _route_after_reasoner(state: AgentState) -> str:
    plan = state.node_states.planner.plan
    status = state.node_states.planner.decision_status
    if status in {DecisionStatus.SATISFIED, DecisionStatus.BLOCKED}:
        return END
    if not plan or not plan.steps or state.node_states.planner.plan_count >= state.max_turns:
        return END
    return EXECUTE_TOOLS_EDGE


def _route_after_execute(state: AgentState) -> str:
    _append_execution_messages(state)
    if state.node_states.planner.plan_count >= state.max_turns:
        state.node_states.planner.decision_status = DecisionStatus.BLOCKED
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
        return END
    return "reasoner"


def _append_execution_messages(state: AgentState) -> None:
    """Preserve the prior tool exchange for the next reasoner invocation."""
    plan = state.node_states.planner.plan
    if plan is None:
        return
    results_by_step = {
        str(result.plan_step_id): result
        for result in state.gather_tool_results()
        if result.plan_step_id is not None
    }
    for step in plan.steps:
        if not step.tool:
            continue
        state.reasoner_message_history.append(AIMessage(
            content="",
            tool_calls=[{
                "id": step.id,
                "type": "tool_call",
                "name": step.tool,
                "args": {
                    "parameters": step.args,
                    "hypothesis": step.hypothesis,
                    **step.metadata,
                },
            }],
        ))
        result = results_by_step.get(str(step.db_id))
        payload = (
            {"tool_name": step.tool, "status": "pending"}
            if result is None
            else bounded_json_value(result.model_dump(mode="json"), max_bytes=12_000)
        )
        state.reasoner_message_history.append(ToolMessage(
            content=json.dumps(payload, default=str),
            tool_call_id=step.id,
            name=step.tool,
        ))


@dataclass(frozen=True)
class MainReasonerStrategy:
    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("reasoner", run_reasoner)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.set_entry_point("reasoner")
        builder.add_conditional_edges(
            "reasoner",
            _route_after_reasoner,
            {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END},
        )
        builder.add_conditional_edges(
            EXECUTE_TOOLS_EDGE,
            _route_after_execute,
            {"reasoner": "reasoner", END: END},
        )
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        final_state = self._compile_graph().invoke(
            agent_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
