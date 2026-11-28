from unittest.mock import Mock

import pytest

from agent_runtime.constants import NodeName, SYNTHESIZE_EDGE
from agent_runtime.context import bind_agent_context
from agent_runtime.execution.tool_executor import _agent_completion_result, run_executor
from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.strategies.react.decision import DecisionResult, record_decision_status
from agent_runtime.strategies.react.graph import ReactStrategy
from agent_runtime.strategies.single_pass.graph import DirectReturnStrategy
from agent_runtime.evaluation.router import evaluator_router
from agent_runtime.evaluation.evaluator import _apply_evaluation
from agent_runtime.models.evaluation_result import EvaluationResult
from agent_runtime.working_memory import (
    MAX_WORKING_MEMORY_BYTES,
    working_memory_path,
)
from agents.delegate.profile import DELEGATE_PROFILE
from agents.main.agent import _route_after_execute, run_reasoner
from capabilities.adapters.workspace.working_memory import update_memory_state
from retrieval.evidence import ToolResult


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("entrypoint", ["react", "main", "direct_return"])
def test_empty_completion_requires_final_memory_save(
    agent_state_factory, plan_factory, monkeypatch, saved, entrypoint,
):
    state = agent_state_factory(profile=DELEGATE_PROFILE)
    state.memory_task_completed = saved
    state.node_states.planner.plan = plan_factory(steps=[])
    if entrypoint == "main":
        monkeypatch.setattr("agents.main.agent.run_decision", lambda *args, **kwargs: DecisionResult(
            plan=plan_factory(steps=[]), status=DecisionStatus.SATISFIED, raw_content="Done",
        ))
        run_reasoner(state)
    else:
        record_decision_status(state, DecisionStatus.SATISFIED)
        route = ReactStrategy()._route_after_decision if entrypoint == "react" else DirectReturnStrategy._route_after_reasoner
        assert route(state) == "__end__"
    assert state.result.answer_status == (AnswerStatus.SUCCESS if saved else AnswerStatus.FAILED)
    assert state.node_states.planner.decision_status == (DecisionStatus.SATISFIED if saved else DecisionStatus.BLOCKED)
    if not saved:
        assert state.result.execution_status == ExecutionStatus.FAILED


def test_non_memory_agent_can_finish_without_memory(agent_state_factory, plan_factory):
    state = agent_state_factory(plan=plan_factory(steps=[]))
    record_decision_status(state, DecisionStatus.SATISFIED)
    assert state.result.answer_status == AnswerStatus.SUCCESS


def test_final_save_at_turn_limit_allows_main_answer(agent_state_factory):
    state = agent_state_factory(profile=DELEGATE_PROFILE)
    state.node_states.planner.plan_count = state.max_turns
    state.memory_task_completed = True
    assert _route_after_execute(state) == NodeName.REASONER.value


def test_evaluator_preserves_success_after_final_save(agent_state_factory):
    state = agent_state_factory(profile=DELEGATE_PROFILE)
    state.memory_task_completed = True
    _apply_evaluation(state, EvaluationResult(status="SATISFIED", relevant_evidence=[]), [])
    assert state.result.answer_status == AnswerStatus.SUCCESS
    assert evaluator_router(state) == SYNTHESIZE_EDGE


def test_direct_return_cannot_skip_final_memory_save(agent_state_factory, plan_factory, plan_step_factory):
    state = agent_state_factory(profile=DELEGATE_PROFILE)
    step = plan_step_factory(tool="web_search")
    state.node_states.planner.plan = plan_factory(steps=[step])
    state.direct_tool_results = [ToolResult(plan_step_id=step.db_id, result={"ok": True})]
    DirectReturnStrategy._route_after_execute(state)
    assert state.result.answer_status == AnswerStatus.FAILED


@pytest.mark.parametrize("next_args", [{"append": "More work"}, {"find": "missing text", "replace": "New"}])
def test_executor_invalidates_final_save_after_more_work(
    working_memory_state_factory, plan_factory, plan_step_factory, next_args,
):
    state = working_memory_state_factory()
    state.node_states.planner.plan = plan_factory(steps=[plan_step_factory(
        tool="update_memory_state", args={"content": "Finished", "task_completed": True},
    )])
    run_executor(state)
    assert state.memory_task_completed
    assert state.result.answer_status == AnswerStatus.SUCCESS

    state.node_states.planner.plan = plan_factory(steps=[plan_step_factory(
        tool="update_memory_state", args=next_args,
    )])
    run_executor(state)
    assert not state.memory_task_completed
    assert not state.can_complete


@pytest.mark.parametrize("mode", ["content", "append", "find"])
@pytest.mark.parametrize("character", ["x", "é"])
def test_oversized_updates_leave_file_and_history_unchanged(
    working_memory_state_factory, monkeypatch, mode, character,
):
    state = working_memory_state_factory(content="Original")
    compact = Mock()
    monkeypatch.setattr(state, "condense_reasoner_history", compact)
    large = character * MAX_WORKING_MEMORY_BYTES + "x"
    args = {mode: large} if mode != "find" else {"find": "Original", "replace": large}
    with bind_agent_context(state), pytest.raises(ValueError, match="consolidate"):
        update_memory_state.invoke(args)
    assert state.read_working_memory() == "Original"
    compact.assert_not_called()


@pytest.mark.parametrize("character", ["x", "é"])
def test_oversized_memory_is_bounded_in_state_prompts_and_handoffs(working_memory_state_factory, character):
    state = working_memory_state_factory(content=character * MAX_WORKING_MEMORY_BYTES)
    # Simulate an externally changed file, bypassing the tool's write validation.
    path = working_memory_path(state)
    path.write_text(character * (MAX_WORKING_MEMORY_BYTES + 1), encoding="utf-8")
    contents = state.read_working_memory()
    assert len(contents.encode("utf-8")) <= MAX_WORKING_MEMORY_BYTES
    assert "truncated" in contents

    prompt = AgentPrompt(instruction="", working_memory=path.read_text(encoding="utf-8"))
    prompt.include_section(PromptSectionKeys.WORKING_MEMORY)
    assert len(prompt.sections_raw[PromptSectionKeys.WORKING_MEMORY].encode("utf-8")) <= MAX_WORKING_MEMORY_BYTES
    assert "truncated" in prompt.build_messages()[0].content

    handoff = _agent_completion_result(agent_name="delegate", state=state, task="Task")
    assert handoff.result["contents"] == contents


def test_memory_handoff_reuses_latest_memory_evidence(working_memory_state_factory, evidence_factory):
    state = working_memory_state_factory()
    memory_evidence = evidence_factory(data={"title": state.working_memory_file_name})
    state.direct_tool_results = [ToolResult(evidence=[memory_evidence])]
    result = _agent_completion_result(agent_name="delegate", state=state, task="Task")
    assert result.evidence == [memory_evidence]
