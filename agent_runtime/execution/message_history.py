from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from common.data import bounded_json_value
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_state import AgentState
from retrieval.evidence import ToolResult


def _model_facing_tool_result(result: ToolResult) -> dict:
    """Serialize tool output without exposing internal execution identifiers.

    Evidence IDs are the only UUIDs that downstream tools should be able to
    reuse. Tool-call and plan-step IDs are transport/debugging identifiers and
    are deliberately kept out of the model-facing tool message.
    """
    payload = {
        "tool_name": result.tool_name,
        "iteration": result.iteration,
        "hypothesis": result.hypothesis,
        "result": bounded_json_value(result.result, max_bytes=12_000),
        "tool_metadata": result.tool_metadata.model_dump(mode="json", exclude_none=True),
        "evidence": [evidence.compact_view() for evidence in result.evidence],
    }
    if result.state is not None:
        payload["state"] = result.state.model_dump(mode="json", exclude_none=True)
    return payload


def append_execution_messages(state: AgentState) -> None:
    """Append the latest plan's tool exchange to the agent's local history."""
    plan = state.node_states.planner.plan
    if plan is None:
        return

    results_by_step = {
        str(result.plan_step_id): result
        for result in state.gather_tool_results()
        if result.plan_step_id is not None
    }
    use_native_tool_args = state.agent_profile.execution_strategy == AgentExecutionStrategy.REACT
    for step in plan.steps:
        if not step.tool and not step.agent:
            continue
        capability_name = step.tool or step.agent
        if step.agent:
            capability_args = {
                "task": step.args.get("task", step.plan),
                "evidence_ids": step.args.get("evidence_ids", []),
            }
        elif use_native_tool_args:
            capability_args = step.args
        else:
            capability_args = {
                "parameters": step.args,
                "hypothesis": step.hypothesis,
                **step.metadata,
            }
        state.reasoner_message_history.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": step.id,
                    "type": "tool_call",
                    "name": capability_name,
                    "args": capability_args,
                }],
            )
        )
        result = results_by_step.get(str(step.db_id))
        payload = (
            {"tool_name": capability_name, "status": "pending"}
            if result is None
            else _model_facing_tool_result(result)
        )
        state.reasoner_message_history.append(
            ToolMessage(
                content=json.dumps(payload, default=str),
                tool_call_id=step.id,
                name=capability_name,
            )
        )
