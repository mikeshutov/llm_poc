from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from common.data import repair_common_json_issues, strip_code_fences
from common.logging import record_node_log
from agent_runtime.constants import EXECUTE_TOOLS_EDGE, NodeName
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.synthesized_result import SynthesisResult, SynthesisResultBlock
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.strategies.react.decision import run_decision
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
            callsite=f"{NodeName.MAIN_REQUEST_STRATEGY.value}.{NodeName.REASONER.value}",
            prompt_builder=build_reasoner_prompt,
            message_history=state.reasoner_message_history,
        )
    except Exception as exc:
        record_node_log(
            roundtrip_id=state.execution_context.roundtrip_id,
            node_name=NodeName.REASONER,
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
    if decision.plan.steps:
        return state
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
    if plan is not None and plan.steps:
        return EXECUTE_TOOLS_EDGE
    if status in {DecisionStatus.SATISFIED, DecisionStatus.BLOCKED}:
        return END
    if not plan or not plan.steps or state.node_states.planner.plan_count >= state.max_turns:
        return END
    return EXECUTE_TOOLS_EDGE


def _route_after_execute(state: AgentState) -> str:
    if state.node_states.planner.plan_count >= state.max_turns:
        state.node_states.planner.decision_status = DecisionStatus.BLOCKED
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
        return END
    return NodeName.REASONER.value


@dataclass(frozen=True)
class MainReasonerStrategy:
    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(NodeName.REASONER.value, run_reasoner)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.set_entry_point(NodeName.REASONER.value)
        builder.add_conditional_edges(
            NodeName.REASONER.value,
            _route_after_reasoner,
            {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END},
        )
        builder.add_conditional_edges(
            EXECUTE_TOOLS_EDGE,
            _route_after_execute,
            {NodeName.REASONER.value: NodeName.REASONER.value, END: END},
        )
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        final_state = self._compile_graph().invoke(
            agent_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
