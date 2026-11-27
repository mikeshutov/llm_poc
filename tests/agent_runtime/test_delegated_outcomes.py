import pytest

from agent_runtime.execution.tool_executor import _agent_completion_result
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from retrieval.evidence import ToolOutcomeStatus


@pytest.mark.parametrize("strategy", list(AgentExecutionStrategy))
@pytest.mark.parametrize("answer, execution, outcome", [
    (AnswerStatus.SUCCESS, ExecutionStatus.SUCCESS, ToolOutcomeStatus.COMPLETED),
    (AnswerStatus.PARTIAL_SUCCESS, ExecutionStatus.SUCCESS, ToolOutcomeStatus.PARTIAL),
    (AnswerStatus.SUCCESS, ExecutionStatus.PARTIAL, ToolOutcomeStatus.PARTIAL),
    (AnswerStatus.FAILED, ExecutionStatus.FAILED, ToolOutcomeStatus.FAILED),
    (AnswerStatus.PENDING, ExecutionStatus.NO_NEW_WORK, ToolOutcomeStatus.FAILED),
    (AnswerStatus.PENDING, ExecutionStatus.PENDING, ToolOutcomeStatus.NEEDS_INPUT),
])
def test_delegated_outcomes_follow_final_status_for_every_strategy(
    agent_state_factory, agent_profile_factory, strategy, answer, execution, outcome,
):
    state = agent_state_factory(profile=agent_profile_factory(
        name="helper", execution_strategy=strategy,
    ))
    state.result = state.result.copy(answer_status=answer, execution_status=execution)
    state.node_states.evaluator.missing_information = ["Missing source"]

    result = _agent_completion_result(agent_name="helper", state=state, task="Research")

    assert result.state.status == outcome
    assert result.state.missing_inputs == ([] if outcome == ToolOutcomeStatus.COMPLETED else ["Missing source"])
    assert result.result["status"] == (
        execution.value if execution not in {ExecutionStatus.PENDING, ExecutionStatus.SUCCESS} else answer.value
    )


def test_delegated_outcome_preserves_failure_reason(agent_state_factory):
    state = agent_state_factory()
    state.result = state.result.copy(answer_status=AnswerStatus.FAILED, execution_status=ExecutionStatus.FAILED)

    result = _agent_completion_result(agent_name="helper", state=state, task="Research", error="Provider failed")

    assert result.state.status == ToolOutcomeStatus.FAILED
    assert result.state.reason == "Provider failed"
