from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from llm.conversation_model_config import PLANNER_STAGE
from agent_runtime.constants import EXECUTE_TOOLS_EDGE, NodeName
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_state import AgentState
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.strategies.react.decision import run_decision
from agent_runtime.strategies.react.prompt import build_react_decision_prompt
from common.logging import record_node_log


def _plan_was_entirely_duplicate_rejected(state: AgentState) -> bool:
    """Return true when every step in the latest plan was an exact duplicate.

    A duplicate is not a recoverable tool error: executing the same request
    again cannot produce new work.  REACT must stop before asking its LLM to
    select the same action again.  Mixed plans are intentionally excluded so
    successful/new work can still be observed and followed up.
    """
    plan = state.node_states.planner.plan
    if plan is None or not plan.steps:
        return False

    results_by_step_id = {
        result.plan_step_id: result
        for result in state.gather_tool_results()
        if result.plan_step_id is not None
    }
    current_results = [results_by_step_id.get(step.db_id) for step in plan.steps]
    if any(result is None for result in current_results):
        return False
    return all(
        isinstance(result.result, dict)
        and result.result.get("status") == "duplicate"
        for result in current_results
    )


@dataclass(frozen=True)
class ReactStrategy:
    @staticmethod
    def _record_decision_status(state: AgentState, status: DecisionStatus) -> None:
        state.node_states.planner.decision_status = status
        if status == DecisionStatus.SATISFIED:
            state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS)
        elif status == DecisionStatus.BLOCKED:
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.FAILED,
            )

    def _decide(self, state: AgentState, *, callsite: str) -> AgentState:
        if state.node_states.planner.plan_count >= state.max_turns:
            return state
        try:
            result = run_decision(
                state, stage=PLANNER_STAGE,
                callsite=callsite,
                prompt_builder=build_react_decision_prompt,
                message_history=state.reasoner_message_history,
            )
        except Exception as exc:
            # Keep the graph's failure semantics, but do not hide failures that
            # occur before an LLM call is recorded. Without this, provider,
            # prompt, or tool-schema errors look like a REACT agent simply
            # skipped its reasoner.
            record_node_log(
                conversation_id=state.execution_context.conversation_id,
                roundtrip_id=state.execution_context.roundtrip_id,
                event_type="reasoner_failure",
                source=state.agent_profile.name,
                agent_name=state.agent_profile.name,
                node_name=NodeName.REASONER,
                agent_version=state.agent_profile.version,
                stage="invoke",
                level="error",
                message="REACT reasoner invocation failed",
                payload={"error": str(exc), "error_type": type(exc).__name__},
            )
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.FAILED,
            )
            state.node_states.planner.plan = None
            return state
        state.node_states.planner.plan = result.plan
        state.node_states.planner.plan_count += 1
        self._record_decision_status(state, result.status)
        return state

    def _reason(self, state: AgentState) -> AgentState:
        return self._decide(
            state,
            callsite=f"{AgentExecutionStrategy.REACT.value}.{NodeName.REASONER.value}",
        )

    def _route_after_decision(self, state: AgentState) -> str:
        plan = state.node_states.planner.plan
        decision_status = state.node_states.planner.decision_status
        if decision_status == DecisionStatus.BLOCKED:
            return END
        if decision_status == DecisionStatus.SATISFIED:
            return END
        if plan is None or not plan.steps or state.node_states.planner.plan_count > state.max_turns:
            if not plan or not plan.steps:
                state.result = state.result.copy(answer_status=AnswerStatus.FAILED)
            return END
        return EXECUTE_TOOLS_EDGE

    def _route_after_execute(self, state: AgentState) -> str:
        if _plan_was_entirely_duplicate_rejected(state):
            state.node_states.planner.decision_status = DecisionStatus.BLOCKED
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.NO_NEW_WORK,
            )
            return END
        if state.node_states.planner.plan_count >= state.max_turns:
            state.node_states.planner.decision_status = DecisionStatus.BLOCKED
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.FAILED,
            )
            return END
        return NodeName.REASONER.value

    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(NodeName.REASONER.value, self._reason)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.set_entry_point(NodeName.REASONER.value)
        builder.add_conditional_edges(NodeName.REASONER.value, self._route_after_decision, {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END})
        builder.add_conditional_edges(EXECUTE_TOOLS_EDGE, self._route_after_execute, {NodeName.REASONER.value: NodeName.REASONER.value, END: END})
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        graph = self._compile_graph()
        final_state = graph.invoke(agent_state, config={"configurable": {"thread_id": thread_id}})
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
