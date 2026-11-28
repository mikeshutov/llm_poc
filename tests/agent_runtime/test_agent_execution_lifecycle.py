import pytest

from agent_runtime.context import current_agent_state
from agent_runtime.execution.tool_executor import execute_agent_tool, run_executor
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from capabilities.adapters.workspace.working_memory import update_memory_state
from capabilities.discovery.models import AvailableTool, CapabilityDiscoveryResult
from capabilities.registry.tools import TOOL_CATEGORIES, WORKING_MEMORY_TOOLS


@pytest.fixture
def memory_delegation(runtime_workspace, delegated_agent_state_factory, agent_profile_factory):
    profile = agent_profile_factory(
        name="delegate", execution_strategy=AgentExecutionStrategy.REACT,
        excluded_categories=set(TOOL_CATEGORIES), static_tools=list(WORKING_MEMORY_TOOLS),
        working_memory_template="# Working Memory\n\n## Objective\n{{ objective }}",
    )
    return delegated_agent_state_factory(
        child_profile=profile, task="Research one subquestion", evidence_ids=["ev-1"],
    )


@pytest.mark.parametrize("reuse_evidence", [False, True])
def test_delegation_initializes_memory_before_discovery_and_returns_saved_state(
    memory_delegation, runtime_workspace, monkeypatch, reuse_evidence,
):
    events = []
    child = memory_delegation.available_agent_states["delegate"]
    evidence = []

    def discover(state, *, include_agents, query, allowed_tool_names):
        events.append("discovery")
        assert current_agent_state.get() is state
        assert not include_agents
        assert allowed_tool_names == child.agent_profile.tool_names
        assert query == state.inputs.task == "Research one subquestion"
        assert (runtime_workspace / state.working_memory_file_name).is_file()
        assert query in state.read_working_memory()
        return CapabilityDiscoveryResult(tools=[AvailableTool(name="math_calculate", category="math")])

    def run_child(state):
        events.append("strategy")
        assert current_agent_state.get() is state
        assert state.inputs.evidence_ids == ["ev-1"]
        assert not state.inputs.discovered_attribute_types
        assert state.inputs.discovered_capabilities.tools[0].name == "math_calculate"
        saved = update_memory_state.invoke({"content": "Finished. Supporting evidence: ev-1"})
        if reuse_evidence:
            state.direct_tool_results.append(saved)
            evidence.extend(saved.evidence)
        state.result = state.result.copy(
            answer_status=AnswerStatus.SUCCESS, execution_status=ExecutionStatus.SUCCESS,
        )
        return state

    monkeypatch.setattr("agent_runtime.execution.tool_executor.capability_discovery", discover)
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: run_child)
    run_executor(memory_delegation)

    result = memory_delegation.direct_tool_results[0]
    assert events == ["discovery", "strategy"]
    assert result.result["status"] == "success"
    assert result.result["contents"] == "Finished. Supporting evidence: ev-1"
    assert result.result["filename"].startswith("memory/")
    assert result.result["error"] is None
    assert len(result.evidence) == 1
    assert result.evidence[0].data["title"] == result.result["filename"]
    if reuse_evidence:
        assert result.evidence[0].id == evidence[0].id
    assert result.plan_step_id == memory_delegation.node_states.planner.plan.steps[0].db_id
    assert child.inputs.task == ""
    assert child.working_memory_file_name is None
    assert current_agent_state.get() is None


@pytest.mark.parametrize("failure_stage", ["discovery", "strategy"])
def test_failed_delegation_preserves_memory_handoff(memory_delegation, monkeypatch, failure_stage):
    def fail(state):
        assert state.working_memory_file_name
        raise RuntimeError("Provider unavailable")

    def discover(state, **kwargs):
        if failure_stage == "discovery":
            return fail(state)
        return CapabilityDiscoveryResult()

    monkeypatch.setattr("agent_runtime.execution.tool_executor.capability_discovery", discover)
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: fail)
    run_executor(memory_delegation)

    result = memory_delegation.direct_tool_results[0]
    assert result.result["status"] == "failed"
    assert result.result["error"] == "Provider unavailable"
    assert "Research one subquestion" in result.result["contents"]
    assert len(result.evidence) == 1
    assert current_agent_state.get() is None


def test_delegation_without_memory_preserves_exception_behavior(
    delegated_agent_state_factory, agent_profile_factory, monkeypatch,
):
    parent = delegated_agent_state_factory(child_profile=agent_profile_factory(name="delegate"))

    def fail(state):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: fail)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        execute_agent_tool(parent, "delegate", {"task": "Do the work"})
    assert current_agent_state.get() is None
