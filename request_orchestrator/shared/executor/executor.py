from __future__ import annotations

from langsmith import traceable
from pydantic import ValidationError
from rendering.debug import build_step_status_message, emit_status_message, prefix_agent_status
from request_orchestrator.models.agent_state import AgentState, IterationState
from tool.registry import call_tool
from tool.repository.tool_call_repository import ToolCallRepository

def _next_step(iteration: IterationState):
    if iteration.plan is None:
        return None
    for step in iteration.plan.steps:
        if step.id not in iteration.results:
            return step
    return None

#OPTIONAL MAY NOT BE WORTH IT
def _substitute_refs(obj, results: dict):
    if isinstance(obj, str):
        if obj.startswith("#E"):
            key = obj[1:]  # "#E1" -> "E1"
            return results.get(key, obj)
        return obj
    if isinstance(obj, list):
        return [_substitute_refs(x, results) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_refs(v, results) for k, v in obj.items()}
    return obj

@traceable(name="Executor Node")
def run_executor(agent_state: AgentState) -> AgentState:
    iteration = agent_state.iteration_trace[-1] 
    tool_repo = ToolCallRepository() if agent_state.roundtrip_id and agent_state.agent_profile.persist_tool_calls else None
    allowed_tool_names = agent_state.agent_profile.allowed_tool_names()

    while (step := _next_step(iteration)) is not None:
        args = _substitute_refs(step.args, iteration.results)

        emit_status_message(prefix_agent_status(agent_state.agent_profile.name, build_step_status_message(step.plan, step.tool, args)))

        try:
            out = call_tool(name=step.tool, tool_input=args, allowed_tool_names=allowed_tool_names)
        except ValidationError as e:
            out = {"error": f"Invalid arguments for tool '{step.tool}': {e.errors(include_url=False)}"}
        except Exception as e:
            out = {"error": f"Tool '{step.tool}' failed: {e}", "tool": step.tool}
        iteration.results[step.id] = out

        if tool_repo and agent_state.roundtrip_id:
            tool_repo.append_tool_call(agent_state.roundtrip_id, iteration, step)

    return agent_state
