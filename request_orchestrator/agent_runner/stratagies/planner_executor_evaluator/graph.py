from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.validator import validator
from request_orchestrator.agent_runner.stratagies.planner_executor_evaluator.result_validator import execution_result_router, run_execution_result_validator
from request_orchestrator.constants import EVALUATE_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.evaluator import evaluator_router
from request_orchestrator.shared.evaluator import run_evaluator
from request_orchestrator.shared.executor.executor import run_executor
from request_orchestrator.shared.planner.planner import run_planner

AgentRouter = Callable[[AgentState], str]


@dataclass(frozen=True)
class PlannerExecutorEvaluatorStratagy:
    execute_router: AgentRouter

    def _compile_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node(PLAN_EDGE, run_planner)
        builder.add_node(EVALUATE_EDGE, run_evaluator)
        builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
        builder.add_node("validate_execution_result", run_execution_result_validator)
        builder.set_entry_point(PLAN_EDGE)

        builder.add_conditional_edges(
            PLAN_EDGE,
            validator,
            {
                EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE,
                SYNTHESIZE_EDGE: END,
            },
        )

        builder.add_conditional_edges(
            EVALUATE_EDGE,
            evaluator_router,
            {
                PLAN_EDGE: PLAN_EDGE,
                SYNTHESIZE_EDGE: END,
            },
        )

        builder.add_edge(EXECUTE_TOOLS_EDGE, "validate_execution_result")
        builder.add_conditional_edges(
            "validate_execution_result",
            execution_result_router,
            {
                PLAN_EDGE: PLAN_EDGE,
                EVALUATE_EDGE: EVALUATE_EDGE,
                SYNTHESIZE_EDGE: END,
                END: END,
            },
        )

        return builder.compile()

    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState:
        graph = self._compile_graph()
        final_state = graph.invoke(
            agent_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
