from importlib import import_module


__all__ = [
    "AgentProfile",
    "AgentPrompt",
    "AgentExecutionContext",
    "AgentNodeStates",
    "AgentResult",
    "AgentState",
    "EvaluatorEventPayload",
    "MainState",
    "Plan",
    "EvidenceStep",
    "PlanStep",
    "PlanningResult",
    "PlannerNodeState",
    "EvaluatorNodeState",
    "RequestAnalysis",
]


_EXPORTS = {
    "AgentProfile": ("request_orchestrator.agent_runner.models.agent_profile", "AgentProfile"),
    "AgentPrompt": ("request_orchestrator.models.agent_prompt", "AgentPrompt"),
    "AgentExecutionContext": ("request_orchestrator.models.agent_execution_context", "AgentExecutionContext"),
    "AgentNodeStates": ("request_orchestrator.shared.node_state", "AgentNodeStates"),
    "EvidenceStep": ("request_orchestrator.models.agent_prompt", "EvidenceStep"),
    "EvaluatorNodeState": ("request_orchestrator.shared.evaluator_state", "EvaluatorNodeState"),
    "AgentResult": ("request_orchestrator.models.agent_result", "AgentResult"),
    "AgentState": ("request_orchestrator.models.agent_state", "AgentState"),
    "EvaluatorEventPayload": ("request_orchestrator.models.evaluator_event_payload", "EvaluatorEventPayload"),
    "RequestAnalysis": ("request_orchestrator.models.request_analysis", "RequestAnalysis"),
    "MainState": ("request_orchestrator.models.main_state", "MainState"),
    "Plan": ("request_orchestrator.models.plan", "Plan"),
    "PlanStep": ("request_orchestrator.models.plan", "PlanStep"),
    "PlanningResult": ("request_orchestrator.models.plan", "PlanningResult"),
    "PlannerNodeState": ("request_orchestrator.shared.planner_state", "PlannerNodeState"),
}


def __getattr__(name: str):
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
