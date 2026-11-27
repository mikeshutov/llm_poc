from dataclasses import replace

import pytest

from agent_runtime.constants import PLAN_EDGE, SYNTHESIZE_EDGE
from agent_runtime.evaluation.router import evaluator_router
from agent_runtime.evaluation.evaluator import _apply_evaluation
from agent_runtime.evaluation.prompts.evaluator_prompt import build_evaluator_prompt
from agent_runtime.execution.tool_executor import _agent_completion_result
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_prompt import PromptSectionKeys
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.evaluation_result import EvaluationResult
from agent_runtime.planning.prompts.planner_prompt import build_planner_prompt
from agent_runtime.strategies.plan_execute_evaluate.graph import PlannerExecutorEvaluatorStrategy
from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt
from agents.main.prompts import build_reasoner_prompt
from agents.main.request import TOP_LEVEL_PROFILE
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from agents.research.profile import RESEARCH_PROFILE
from capabilities.registry.constants import TOOL_CATEGORY_MATH
from capabilities.registry.global_registry import call_tool
from personalization.profile.models import UserProfile, UserAttributesSection
from retrieval.evidence import ToolResult


@pytest.mark.parametrize("evaluation", ["SATISFIED", "PARTIAL_BUT_SUFFICIENT"])
def test_sufficient_evidence_requires_a_final_state_save(agent_state_factory, evaluation):
    state = agent_state_factory(profile=RESEARCH_PROFILE)
    state.node_states.evaluator.evaluation_status = evaluation

    assert evaluator_router(state) == PLAN_EDGE


@pytest.mark.parametrize("evaluation", ["SATISFIED", "PARTIAL_BUT_SUFFICIENT"])
def test_turn_limit_without_final_save_is_not_success(agent_state_factory, evaluation):
    state = agent_state_factory(profile=RESEARCH_PROFILE)
    state.node_states.planner.plan_count = state.max_turns
    state.node_states.evaluator.evaluation_status = evaluation
    state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS, execution_status=ExecutionStatus.SUCCESS)
    _apply_evaluation(state, EvaluationResult(status=evaluation, relevant_evidence=[]), [])

    assert evaluator_router(state) == SYNTHESIZE_EDGE
    assert state.result.answer_status == AnswerStatus.FAILED
    assert state.result.execution_status == ExecutionStatus.FAILED


def test_agents_without_memory_keep_evaluator_completion(agent_state_factory):
    profile = replace(RESEARCH_PROFILE, working_memory_template="")
    state = agent_state_factory(profile=profile)
    state.node_states.evaluator.evaluation_status = "SATISFIED"

    assert evaluator_router(state) == SYNTHESIZE_EDGE


@pytest.mark.parametrize("evaluation", ["RETRYABLE", "TERMINAL", "SATISFIED"])
@pytest.mark.parametrize("answer, execution, expected", [
    (AnswerStatus.SUCCESS, ExecutionStatus.SUCCESS, "completed"),
    (AnswerStatus.FAILED, ExecutionStatus.FAILED, "failed"),
    (AnswerStatus.PARTIAL_SUCCESS, ExecutionStatus.PARTIAL, "partial"),
])
def test_delegated_outcome_uses_runtime_result_not_evaluator(
    agent_state_factory,
    evaluation,
    answer,
    execution,
    expected,
):
    state = agent_state_factory(profile=RESEARCH_PROFILE)
    state.result = state.result.copy(answer_status=answer, execution_status=execution)
    state.node_states.evaluator.evaluation_status = evaluation
    state.node_states.evaluator.missing_information = ["Old evaluator gap"]

    result = _agent_completion_result(agent_name="research", state=state, task="Research")

    assert result.state.status == expected
    if expected == "completed":
        assert result.state.missing_inputs == []
        assert result.state.reason is None


@pytest.mark.parametrize("strategy", list(AgentExecutionStrategy))
def test_database_agents_are_not_instructed_to_use_unavailable_memory(
    db_agent_factory,
    agent_state_factory,
    strategy,
):
    profile = db_agent_factory(
        user_id="alice", name="database_helper", planner_instruction="Complete the assigned task.",
        execution_strategy=strategy, allowed_categories=[TOOL_CATEGORY_MATH],
    ).to_agent_profile()
    state = agent_state_factory(profile=profile)
    builder = build_planner_prompt if strategy == AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR else build_subagent_decision_prompt

    prompt, tools = builder(state)

    assert not profile.uses_memory_state
    assert "update_memory_state" not in {tool.name for tool in tools}
    assert "update_memory_state" not in prompt.build()
    assert "task_completed=true" not in prompt.build()


def test_profile_management_has_no_impossible_memory_instructions(agent_state_factory):
    state = agent_state_factory(profile=PROFILE_MANAGEMENT_PROFILE)
    prompt, _ = build_subagent_decision_prompt(state)

    assert "update_memory_state" not in prompt.build()
    assert "task_completed=true" not in prompt.build()


@pytest.mark.parametrize("builder", [build_planner_prompt, lambda state: (build_evaluator_prompt(state, []), [])])
def test_research_prompts_exclude_profile_fields_and_attribute_types(
    user_attribute_factory,
    agent_state_factory,
    builder,
):
    state = agent_state_factory(profile=RESEARCH_PROFILE)
    state.execution_context.user_profile = UserProfile(
        display_name="PRIVATE_NAME",
        user_attributes=UserAttributesSection(attributes=[user_attribute_factory(
            user_id="alice",
            value=["PRIVATE_VALUE"],
            attribute_type="PRIVATE_ATTRIBUTE_TYPE",
            source=None,
            created_at="",
            updated_at="",
            confidence=None,
            importance=None,
        )]),
    )

    prompt, _ = builder(state)

    assert prompt.user_profile is None
    assert PromptSectionKeys.USER_PROFILE not in prompt.sections_raw
    assert "PRIVATE_" not in prompt.build()
    assert all("PRIVATE_" not in message.content for message in prompt.build_messages())


def test_main_reasoner_still_receives_user_profile(agent_state_factory):
    state = agent_state_factory(profile=TOP_LEVEL_PROFILE)
    state.execution_context.user_profile = UserProfile(display_name="Main user")

    prompt, _ = build_reasoner_prompt(state)

    assert PromptSectionKeys.USER_PROFILE in prompt.sections_raw
    assert "Main user" in prompt.build()


@pytest.mark.parametrize("save_fails", [False, True])
def test_research_graph_saves_final_memory_after_evaluator_says_sufficient(
    evidence_factory,
    plan_factory,
    plan_step_factory,
    monkeypatch,
    working_memory_state_factory,
    save_fails,
):
    state = working_memory_state_factory(
        profile=RESEARCH_PROFILE, profile_overrides={"max_turns": 3}, task="Research the assigned topic",
    )
    events = []

    def plan(current):
        current.node_states.planner.plan_count += 1
        turn = current.node_states.planner.plan_count
        if turn == 1:
            steps = [plan_step_factory(id="search", tool="web_search", plan="Research", args={})]
        elif turn == 2:
            steps = []  # Ask the evaluator whether the evidence is sufficient.
        else:
            assert turn == 3
            assert events == ["search", "evaluate"]
            update_args = {"find": "missing text", "replace": "updated"} if save_fails else {
                "content": "Final checked findings: evidence-1", "task_completed": True,
            }
            steps = [plan_step_factory(
                id="finish",
                tool="update_memory_state",
                plan="Save final findings",
                args=update_args,
            )]
        current.node_states.planner.plan = plan_factory(steps=steps)
        return current

    def evaluate(current):
        events.append("evaluate")
        _apply_evaluation(current, EvaluationResult(status="SATISFIED", relevant_evidence=[]), [])
        return current

    def execute(name, **kwargs):
        if name == "update_memory_state":
            events.append("save")
            return call_tool(name=name, **kwargs)
        events.append("search")
        return ToolResult(evidence=[evidence_factory(
            id="evidence-1",
            type="test",
            source="test",
            summary="Checked findings",
        )])

    monkeypatch.setattr("agent_runtime.strategies.plan_execute_evaluate.graph.run_planner", plan)
    monkeypatch.setattr("agent_runtime.strategies.plan_execute_evaluate.graph.run_evaluator", evaluate)
    monkeypatch.setattr("agent_runtime.execution.tool_executor.call_tool", execute)

    completed = PlannerExecutorEvaluatorStrategy(execute_router=lambda state: PLAN_EDGE).run(state, thread_id="test")

    if save_fails:
        assert events == ["search", "evaluate", "save", "evaluate"]
        assert "Final checked findings" not in completed.read_working_memory()
        assert completed.result.answer_status == AnswerStatus.FAILED
        assert completed.result.execution_status == ExecutionStatus.FAILED
    else:
        assert events == ["search", "evaluate", "save"]
        assert completed.read_working_memory() == "Final checked findings: evidence-1"
        assert completed.result.answer_status == AnswerStatus.SUCCESS
        assert completed.result.execution_status == ExecutionStatus.SUCCESS
