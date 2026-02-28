from agent.context import AgentContext
from agent.models.agent_result import AgentResult
from agent.models.tool_call_status import ToolCallStatus
from agent.routes.finalize_result import finalize_result
from agent.runtime.runtime_utils import unsupported_response
from agent.runtime.trace_service import trace_and_advance
from agent.tools.tool_name import ToolName


def handle_unsupported_route(
    *,
    ctx: AgentContext,
) -> AgentResult:
    state = ctx.state
    unsupported = unsupported_response()
    trace_and_advance(
        state,
        turn_index=0,
        tool_name=ToolName.UNKNOWN_INTENT_HANDLER.value,
        status=ToolCallStatus.SUCCESS.value,
        reason=ctx.decision.reason,
        input_payload={"intent": str(ctx.parsed_query.intent)},
        output_payload={"response": unsupported.response},
        goal=state.loop_state.goal,
        done=True,
        duration_ms=0,
    )
    return finalize_result(
        ctx=ctx,
        answer=unsupported,
        goal=state.loop_state.goal,
    )
