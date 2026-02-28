from agent.context import AgentContext
from agent.models.agent_result import AgentResult
from agent.runtime.runtime_utils import build_debug_trace
from response_layer.models.response_payload import ResponsePayload


def finalize_result(
    *,
    ctx: AgentContext,
    answer: ResponsePayload,
    goal: str,
) -> AgentResult:
    return AgentResult(
        answer=answer,
        debug_trace=build_debug_trace(
            decision=ctx.decision,
            parsed_intent=str(ctx.parsed_query.intent),
            max_turns=ctx.max_turns,
            iterations_used=ctx.state.iterations_used,
            goal=goal,
            goal_reached=ctx.state.loop_state.goal_reached,
            traces=ctx.state.tool_traces,
        ),
        tool_traces=ctx.state.tool_traces,
        goal=goal,
        goal_reached=ctx.state.loop_state.goal_reached,
        iterations_used=ctx.state.iterations_used,
    )
