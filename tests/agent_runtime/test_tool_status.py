import pytest
from agent_runtime.working_memory import task_completion_requested
from retrieval.evidence import ToolOutcome, ToolOutcomeStatus, ToolResult


@pytest.mark.parametrize("payload, outcome, status", [
    ({"task_completed": True}, None, "completed"),
    ({"status": "failed", "task_completed": True}, None, "error"),
    ({"error": "write failed", "task_completed": True}, None, "error"),
    ({"error_message": "write failed", "task_completed": True}, None, "error"),
    ({"status": "rejected", "task_completed": True}, None, "rejected"),
    ({"status": "duplicate", "task_completed": True}, None, "rejected"),
    ({"task_completed": True}, ToolOutcome(status=ToolOutcomeStatus.FAILED), "error"),
])
def test_memory_completion_uses_common_result_status(payload, outcome, status):
    result = ToolResult(tool_name="update_memory_state", result=payload, state=outcome)

    assert result.status == status
    assert task_completion_requested(result) is (status == "completed")
