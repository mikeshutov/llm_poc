from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from agent_runtime.strategies.plan_execute_evaluate.validator import validator
from agent_runtime.strategies.plan_execute_evaluate.result_validator import execution_result_router, run_execution_result_validator
from agent_runtime.constants import DISCOVER_AGENT_STATE_EDGE, EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE, NodeName
from agent_runtime.models.agent_state import AgentState
from agent_runtime.evaluation import evaluator_router, run_evaluator
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.planning.planner import run_planner
from agent_runtime.state_enrichment import discover_agent_state_node

AgentRouter = Callable[[AgentState], str]


@dataclass(frozen=True)
class PlannerExecutorEvaluatorStrategy:
    execute_router: AgentRouter

    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(DISCOVER_AGENT_STATE_EDGE, discover_agent_state_node)
        builder.add_node(PLAN_EDGE, run_planner)
        builder.add_node(EVALUATE_EDGE, run_evaluator)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.add_node(NodeName.VALIDATE_EXECUTION_RESULT.value, run_execution_result_validator)
        builder.set_entry_point(PLAN_EDGE)
        builder.add_conditional_edges(PLAN_EDGE, validator, {
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
            DISCOVER_AGENT_STATE_EDGE: DISCOVER_AGENT_STATE_EDGE,
            SYNTHESIZE_EDGE: END,
        })
        builder.add_conditional_edges(EVALUATE_EDGE, evaluator_router, {
            PLAN_EDGE: DISCOVER_AGENT_STATE_EDGE,
            SYNTHESIZE_EDGE: END,
        })
        builder.add_edge(EXECUTE_TOOLS_EDGE, NodeName.VALIDATE_EXECUTION_RESULT.value)
        builder.add_conditional_edges(NodeName.VALIDATE_EXECUTION_RESULT.value, execution_result_router, {
            PLAN_EDGE: DISCOVER_AGENT_STATE_EDGE,
            EVALUATE_EDGE: EVALUATE_EDGE,
            SYNTHESIZE_EDGE: END,
            END: END,
        })
        builder.add_edge(DISCOVER_AGENT_STATE_EDGE, PLAN_EDGE)
        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        graph = self._compile_graph()
        final_state = graph.invoke(agent_state, config={"configurable": {"thread_id": thread_id}})
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
