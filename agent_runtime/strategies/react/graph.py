from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from llm.conversation_model_config import PLANNER_STAGE
from agent_runtime.constants import DISCOVER_AGENT_STATE_EDGE, EXECUTE_TOOLS_EDGE, NodeName
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_state import AgentState
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.state_enrichment import discover_agent_state_node
from agent_runtime.strategies.react.decision import DecisionStatus, run_decision
from agent_runtime.strategies.react.prompt import build_react_decision_prompt


@dataclass(frozen=True)
class ReactStrategy:
    @staticmethod
    def _record_decision_status(state: AgentState, status: str) -> None:
        normalized = status.upper()
        if normalized in {"DONE", "COMPLETE", "COMPLETED"}:
            normalized = DecisionStatus.SATISFIED
        state.node_states.planner.decision_status = normalized
        if normalized == DecisionStatus.SATISFIED:
            state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS)
        elif normalized == DecisionStatus.BLOCKED:
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

    def _reason(self, state: AgentState) -> AgentState:
        return self._decide(state, callsite="react.reason")

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
        builder.add_node(DISCOVER_AGENT_STATE_EDGE, discover_agent_state_node)
        builder.add_node(NodeName.REASONER.value, self._reason)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.set_entry_point(NodeName.REASONER.value)
        builder.add_conditional_edges(NodeName.REASONER.value, self._route_after_decision, {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END})
        builder.add_conditional_edges(EXECUTE_TOOLS_EDGE, self._route_after_execute, {NodeName.REASONER.value: DISCOVER_AGENT_STATE_EDGE, END: END})
        builder.add_edge(DISCOVER_AGENT_STATE_EDGE, NodeName.REASONER.value)
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        graph = self._compile_graph()
        final_state = graph.invoke(agent_state, config={"configurable": {"thread_id": thread_id}})
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
