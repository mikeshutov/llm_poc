import pytest

from agent_runtime.execution.tool_executor import execute_step
from retrieval.evidence import ToolOutcome, ToolOutcomeStatus, ToolResult


@pytest.mark.parametrize("payload, outcome, status", [
    ({}, None, "completed"),
    (None, None, "completed"),
    ({"status": "failed"}, None, "error"),
    ({"status": "error"}, None, "error"),
    ({"error": "write failed"}, None, "error"),
    ({"error_message": "write failed"}, None, "error"),
    ({"status": "rejected"}, None, "rejected"),
    ({"status": "duplicate"}, None, "rejected"),
    ({}, ToolOutcome(status=ToolOutcomeStatus.FAILED), "error"),
])
def test_tool_result_status(payload, outcome, status):
    assert ToolResult(result=payload, state=outcome).status == status


def test_execute_step_preserves_error_message_and_execution_ids(monkeypatch, plan_step_factory):
    step = plan_step_factory(tool="tool_a")
    monkeypatch.setattr(
        "agent_runtime.execution.tool_executor.call_tool",
        lambda **kwargs: ToolResult(result={"error_message": "provider failed"}),
    )

    execution = execute_step(step, tool_results_by_step_id={}, allowed_tool_names={"tool_a"})

    assert execution.status == "error"
    assert execution.error_text == "provider failed"
    assert execution.output.tool_name == step.tool
    assert execution.output.plan_step_id == step.db_id
