from __future__ import annotations

from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_inputs import AgentInputs
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.strategies.planner_executor_evaluator.graph import PlannerExecutorEvaluatorStratagy
from request_orchestrator.shared.main_router import router


TOP_LEVEL_PROFILE = AgentProfile(
    name="request_orchestrator",
    scope=MAIN_AGENT_MODEL_SCOPE,
    description="Top-level request planner and executor.",
)
TOP_LEVEL_STRATEGY = PlannerExecutorEvaluatorStratagy(router)


def run_main_request_strategy(main_state: MainState) -> MainState:
    execution_state = AgentState.new(
        agent_profile=TOP_LEVEL_PROFILE,
        inputs=AgentInputs.new(task=main_state.task, request_task=main_state.task),
        execution_context=main_state.execution_context,
        llm=main_state.llm,
    )
    execution_state.inputs.discovered_capabilities = main_state.discovered_capabilities
    execution_state.available_agent_states = main_state.agent_states
    execution_state = TOP_LEVEL_STRATEGY.run(
        execution_state,
        thread_id=main_state.execution_context.conversation_id or "",
    )
    main_state.main_agent_state = execution_state
    main_state.relevant_evidence_ids = list(execution_state.result.relevant_evidence_ids)
    main_state.plan = execution_state.node_states.planner.plan
    main_state.evaluation_status = execution_state.node_states.evaluator.evaluation_status
    main_state.missing_information = list(execution_state.node_states.evaluator.missing_information)
    return main_state
