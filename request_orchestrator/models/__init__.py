from importlib import import_module


__all__ = [
    "AgentProfile",
    "AgentPrompt",
    "AgentResult",
    "AgentState",
    "IterationState",
    "MainState",
    "Plan",
    "EvidenceStep",
    "PlanStep",
    "PlanningResult",
    "PreviousIteration",
    "PreviousIterationStep",
    "RequestAnalysis",
]


_EXPORTS = {
    "AgentProfile": ("request_orchestrator.models.agent_profile", "AgentProfile"),
    "AgentPrompt": ("request_orchestrator.models.agent_prompt", "AgentPrompt"),
    "EvidenceStep": ("request_orchestrator.models.agent_prompt", "EvidenceStep"),
    "PreviousIteration": ("request_orchestrator.models.agent_prompt", "PreviousIteration"),
    "PreviousIterationStep": ("request_orchestrator.models.agent_prompt", "PreviousIterationStep"),
    "AgentResult": ("request_orchestrator.models.agent_result", "AgentResult"),
    "AgentState": ("request_orchestrator.models.agent_state", "AgentState"),
    "IterationState": ("request_orchestrator.models.agent_state", "IterationState"),
    "RequestAnalysis": ("request_orchestrator.models.agent_state", "RequestAnalysis"),
    "MainState": ("request_orchestrator.models.main_state", "MainState"),
    "Plan": ("request_orchestrator.models.plan", "Plan"),
    "PlanStep": ("request_orchestrator.models.plan", "PlanStep"),
    "PlanningResult": ("request_orchestrator.models.plan", "PlanningResult"),
}


def __getattr__(name: str):
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
