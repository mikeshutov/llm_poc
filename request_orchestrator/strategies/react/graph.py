from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from llm.conversation_model_config import PLANNER_STAGE
from request_orchestrator.constants import DISCOVER_AGENT_STATE_EDGE, EXECUTE_TOOLS_EDGE
from request_orchestrator.models.agent_result import ResultStatus
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.state_enrichment import discover_agent_state_node
from request_orchestrator.strategies.react.decision import run_decision
from request_orchestrator.strategies.react.prompt import build_react_decision_prompt


@dataclass(frozen=True)
class ReactStratagy:
    def _initial_decision(self, state: AgentState) -> AgentState:
        try:
            result = run_decision(
                state, stage=PLANNER_STAGE,
                callsite="react.initial_decision",
                prompt_builder=build_react_decision_prompt,
            )
        except Exception:
            state.result = state.result.copy(result_status=ResultStatus.FAILED)
            state.node_states.planner.plan = None
            return state
        state.node_states.planner.plan = result.plan
        state.node_states.planner.plan_count += 1
        return state

    def _reason(self, state: AgentState) -> AgentState:
        if state.node_states.planner.plan_count >= state.max_turns:
            return state
        try:
            result = run_decision(
                state, stage=PLANNER_STAGE,
                callsite="react.planner_decision",
                prompt_builder=build_react_decision_prompt,
            )
        except Exception:
            state.result = state.result.copy(result_status=ResultStatus.FAILED)
            state.node_states.planner.plan = None
            return state
        state.node_states.planner.plan = result.plan
        state.node_states.planner.plan_count += 1
        return state

    def _route_after_decision(self, state: AgentState) -> str:
        plan = state.node_states.planner.plan
        if plan is None or not plan.steps or state.node_states.planner.plan_count > state.max_turns:
            return END
        return EXECUTE_TOOLS_EDGE

    def _route_after_execute(self, state: AgentState) -> str:
        if state.node_states.planner.plan_count >= state.max_turns:
            return END
        return "reason"

    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(DISCOVER_AGENT_STATE_EDGE, discover_agent_state_node)
        builder.add_node("initial_decision", self._initial_decision)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.add_node("reason", self._reason)
        builder.set_entry_point("initial_decision")
        builder.add_conditional_edges("initial_decision", self._route_after_decision, {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END})
        builder.add_conditional_edges(EXECUTE_TOOLS_EDGE, self._route_after_execute, {"reason": DISCOVER_AGENT_STATE_EDGE, END: END})
        builder.add_edge(DISCOVER_AGENT_STATE_EDGE, "reason")
        builder.add_conditional_edges("reason", self._route_after_decision, {EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, END: END})
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        graph = self._compile_graph()
        final_state = graph.invoke(agent_state, config={"configurable": {"thread_id": thread_id}})
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
