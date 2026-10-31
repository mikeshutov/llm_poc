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


@dataclass(frozen=True)
class DirectReturnStrategy:
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

    def _reasoner(self, state: AgentState) -> AgentState:
        if state.node_states.planner.plan_count >= state.max_turns:
            return state
        try:
            result = run_decision(
                state,
                stage=PLANNER_STAGE,
                callsite=f"{AgentExecutionStrategy.DIRECT_RETURN.value}.{NodeName.REASONER.value}",
                prompt_builder=build_react_decision_prompt,
            )
        except Exception:
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

    @staticmethod
    def _route_after_reasoner(state: AgentState) -> str:
        plan = state.node_states.planner.plan
        decision_status = state.node_states.planner.decision_status
        if decision_status in {DecisionStatus.BLOCKED, DecisionStatus.SATISFIED}:
            return END
        if plan is None or not plan.steps or state.node_states.planner.plan_count >= state.max_turns:
            if not plan or not plan.steps:
                state.result = state.result.copy(answer_status=AnswerStatus.FAILED)
            return END
        return EXECUTE_TOOLS_EDGE

    @staticmethod
    def _route_after_execute(state: AgentState) -> str:
        plan = state.node_states.planner.plan
        if plan is None or not plan.steps:
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.FAILED,
            )
            return END

        step_ids = {step.db_id for step in plan.steps}
        calls = [call for call in state.gather_tool_calls() if call.plan_step_id in step_ids]
        if calls and all(call.status == "completed" for call in calls):
            state.result = state.result.copy(
                answer_status=AnswerStatus.SUCCESS,
                execution_status=ExecutionStatus.SUCCESS,
            )
        else:
            state.result = state.result.copy(
                answer_status=AnswerStatus.FAILED,
                execution_status=ExecutionStatus.FAILED,
            )
        return END

    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(NodeName.REASONER.value, self._reasoner)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.set_entry_point(NodeName.REASONER.value)
        builder.add_conditional_edges(
            NodeName.REASONER.value,
            self._route_after_reasoner,
            {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END},
        )
        builder.add_conditional_edges(EXECUTE_TOOLS_EDGE, self._route_after_execute, {END: END})
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        final_state = self._compile_graph().invoke(
            agent_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
