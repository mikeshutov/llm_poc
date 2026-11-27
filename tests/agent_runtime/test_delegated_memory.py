from pathlib import Path

import pytest

from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.context import current_agent_state
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.strategies.react.decision import DecisionResult
from agents.delegate.profile import DELEGATE_PROFILE
from agents.research.profile import RESEARCH_PROFILE
from capabilities.discovery.models import AvailableTool, CapabilityDiscoveryResult
from capabilities.registry.global_registry import available_agent_tools


@pytest.fixture
def delegated_update_memory_state(agent_profile_factory, plan_step_factory, plan_factory, runtime_workspace, agent_state_factory):
    context = AgentExecutionContext.new(conversation_id="conversation-test")
    parent = agent_state_factory(
        profile=agent_profile_factory(name="coordinator", scope="main_agent"), execution_context=context,
        plan=plan_factory(steps=[plan_step_factory(
            id="E1", tool="delegate", plan="Research a bounded subquestion",
            args={"task": "Research one subquestion", "evidence_ids": ["ev-1"]},
        )]),
    )
    parent.available_agent_states = {
        "delegate": agent_state_factory(profile=DELEGATE_PROFILE, execution_context=context),
    }
    return parent


def test_delegation_discovers_then_runs_with_memory_and_signals_completion(
    plan_factory,
    plan_step_factory,
    delegated_update_memory_state,
    tmp_path,
    monkeypatch,
):
    events = []

    def discover(state, *, include_agents, query, allowed_tool_names):
        events.append("discovery")
        assert current_agent_state.get().agent_profile is DELEGATE_PROFILE
        assert current_agent_state.get().execution_context is state.execution_context
        assert not include_agents
        assert allowed_tool_names == DELEGATE_PROFILE.tool_names
        assert query == "Research one subquestion"
        assert state.inputs.task == query
        return CapabilityDiscoveryResult(tools=[AvailableTool(name="math_calculate", category="math")])

    def decide(state, **kwargs):
        events.append("llm")
        assert current_agent_state.get().agent_profile is DELEGATE_PROFILE
        assert state.inputs.discovered_capabilities.tools[0].name == "math_calculate"
        assert not state.inputs.discovered_attribute_types
        assert state.inputs.evidence_ids == ["ev-1"]
        assert (tmp_path / state.working_memory_file_name).is_file()
        assert "Research one subquestion" in state.read_working_memory()
        return DecisionResult(status=DecisionStatus.CONTINUE, plan=plan_factory(steps=[plan_step_factory(
            id="M1", tool="update_memory_state", plan="Save the completed task",
            args={"content": "# Working Memory\nFinished. Supporting evidence: ev-1", "task_completed": True},
        )]))

    monkeypatch.setattr("agent_runtime.execution.tool_executor.capability_discovery", discover)
    monkeypatch.setattr("agent_runtime.strategies.react.graph.run_decision", decide)

    run_executor(delegated_update_memory_state)

    result = delegated_update_memory_state.direct_tool_results[0]
    assert events == ["discovery", "llm"], result.model_dump(mode="json")
    assert result.result["status"] == "success"
    assert result.result["contents"] == "# Working Memory\nFinished. Supporting evidence: ev-1"
    assert result.result["filename"].startswith("memory/")
    assert result.result["error"] is None
    assert len(result.evidence) == 1
    assert result.evidence[0].data["title"] == result.result["filename"]
    assert result.plan_step_id == delegated_update_memory_state.node_states.planner.plan.steps[0].db_id
    assert delegated_update_memory_state.available_agent_states["delegate"].inputs.task == ""


@pytest.mark.parametrize("failure_stage", ["discovery", "strategy"])
def test_failed_delegate_preserves_its_memory_handoff(
    delegated_update_memory_state,
    monkeypatch,
    failure_stage,
):
    monkeypatch.setattr(
        "agent_runtime.execution.tool_executor.capability_discovery",
        lambda state, **kwargs: CapabilityDiscoveryResult(),
    )

    def fail(state):
        assert Path(state.working_memory_file_name).name.startswith(".agent-working-memory-delegate-")
        raise RuntimeError("Provider unavailable")

    if failure_stage == "discovery":
        def fail_discovery(state, **kwargs):
            raise RuntimeError("Provider unavailable")

        monkeypatch.setattr("agent_runtime.execution.tool_executor.capability_discovery", fail_discovery)
        monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: pytest.fail("Strategy must not run"))
    else:
        monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: fail)
    run_executor(delegated_update_memory_state)

    result = delegated_update_memory_state.direct_tool_results[0]
    assert result.result["status"] == "failed"
    assert result.result["error"] == "Provider unavailable"
    assert "failed to complete" in result.result["response"]
    assert "Research one subquestion" in result.result["contents"]
    assert len(result.evidence) == 1


def test_research_retains_its_default_delegate(
    plan_factory,
    plan_step_factory,
    agent_profile_factory,
    agent_state_factory,
    monkeypatch,
):
    parent = agent_state_factory(profile=agent_profile_factory(name="coordinator", scope="main_agent"))
    research = agent_state_factory(profile=RESEARCH_PROFILE)
    delegate = agent_state_factory(profile=DELEGATE_PROFILE)
    parent.available_agent_states = {"research": research, "delegate": delegate}
    parent.node_states.planner.plan = plan_factory(steps=[plan_step_factory(
        id="E1", tool="research", plan="Research the topic", args={"task": "Research a topic"},
    )])
    seen = []

    def run_child(state):
        seen.extend(tool.name for tool in available_agent_tools(state))
        return state

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: run_child)
    run_executor(parent)

    assert "delegate" in seen
    assert "research" not in seen
    assert research.inputs.task == ""
