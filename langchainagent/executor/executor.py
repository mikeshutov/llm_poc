from __future__ import annotations

from langsmith import traceable

from langchainagent.agentstate.model import AgentState, IterationState, PlanStep
from langchainagent.tool_registry import call_tool

def _next_step(state: IterationState) -> PlanStep | None:
    for step in state.steps:
        if step.ref not in state.results:
            return step
    return None

def _substitute_args(arg_text: str, results: dict[str, str]) -> str:
    out = arg_text
    for k, v in results.items():
        if k in out:
            out = out.replace(k, str(v))
    return out

@traceable(name="Executor Node")
def run_executor(state: AgentState) -> AgentState:
    if not state.iteration_trace:
        return state

    iteration = state.iteration_trace[-1]
    step = _next_step(iteration)
    while step is not None:
        tool_input = _substitute_args(step.args, iteration.results)
        out = call_tool(name=step.tool, tool_input=tool_input)
        iteration.results[step.ref] = out
        step = _next_step(iteration)

    state.goal_reached = True
    return state
