import pytest

from agent_runtime.models.plan import PlanStep
from agent_runtime.planning.tool_wrapper import build_tools
from capabilities.registry.global_registry import (
    GLOBAL_TOOL_REGISTRY, available_agent_tools, register_agent_tools, register_default_tools,
)
from agent_runtime.execution.tool_executor import execute_step
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE, DocumentWritingDelegationArgs
from capabilities.discovery.models import AvailableAgent, CapabilityDiscoveryResult


@pytest.fixture(autouse=True)
def initialize_registry():
    register_default_tools()


def test_agent_tool_receives_explicit_task_and_normalizes_evidence(monkeypatch) -> None:
    parent = AgentState.new(
        agent_profile=AgentProfile(name="parent", scope="main_agent"),
        llm=object(),
    )
    child = AgentState.new(agent_profile=DOCUMENT_WRITING_PROFILE, llm=object())
    parent.available_agent_states = {"document_writing": child}
    seen = []

    def run_child(state):
        seen.append(state)
        return state

    monkeypatch.setattr("agents.registry.agent_registry.get", lambda profile: run_child)
    step = PlanStep(
        id="E1", tool="document_writing", plan="Expected outcome",
        hypothesis="Expected outcome",
        args={"task": "Create a document.", "evidence_ids": ["ev-1", " ev-1 ", "ev-2"]},
    )
    result = execute_step(step, agent_state=parent)
    assert seen[0].inputs.task == "Create a document."
    assert seen[0].inputs.evidence_ids == ["ev-1", "ev-2"]
    assert child.inputs.task == ""
    assert result.error_text == ""


def test_agent_delegation_tool_keeps_explicit_task_schema() -> None:
    state = AgentState.new(
        agent_profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    tool = build_tools([GLOBAL_TOOL_REGISTRY.get(state.agent_profile.name)])[0]

    assert list(tool.args_schema.model_fields) == ["parameters", "hypothesis", "evaluation_fields"]
    assert tool.args_schema.model_fields["parameters"].annotation is DocumentWritingDelegationArgs


def test_agent_delegation_tool_accepts_evidence_ids() -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="document_writing", scope="main_agent"),
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    tool = build_tools([GLOBAL_TOOL_REGISTRY.get(state.agent_profile.name)])[0]

    parsed = tool.args_schema.model_fields["parameters"].annotation.model_validate({
        "task": "Create a research paper.",
        "evidence_ids": ["evidence-1", "evidence-1", "  evidence-2  "],
    })

    assert parsed.task == "Create a research paper."
    assert parsed.evidence_ids == ["evidence-1", "evidence-1", "  evidence-2  "]


def test_delegated_agent_states_honor_delegation_policy() -> None:
    parent = AgentState.new(
        agent_profile=AgentProfile(
            name="coordinator",
            scope="main_agent",
            default_available_agent_names={"default_agent"},
        ),
        llm=object(),
    )
    default_agent = AgentState.new(
        agent_profile=AgentProfile(name="default_agent", scope="main_agent"),
        llm=object(),
    )
    undiscovered_agent = AgentState.new(
        agent_profile=AgentProfile(name="undiscovered_agent", scope="main_agent"),
        llm=object(),
    )
    blocked_agent = AgentState.new(
        agent_profile=AgentProfile(name="blocked_agent", scope="main_agent", delegatable=False),
        llm=object(),
    )
    parent.available_agent_states = {
        "coordinator": parent,
        "default_agent": default_agent,
        "undiscovered_agent": undiscovered_agent,
        "blocked_agent": blocked_agent,
    }
    parent.inputs.discovered_capabilities = CapabilityDiscoveryResult(
        agents=[AvailableAgent(agent="default_agent")]
    )

    register_agent_tools([default_agent.agent_profile])
    visible = available_agent_tools(parent)

    assert [tool.name for tool in visible] == ["default_agent"]


def test_non_delegating_agent_exposes_no_delegation_targets() -> None:
    parent = AgentState.new(
        agent_profile=AgentProfile(
            name="leaf",
            scope="main_agent",
            can_delegate_agents=False,
        ),
        llm=object(),
    )
    parent.available_agent_states = {
        "other": AgentState.new(
            agent_profile=AgentProfile(name="other", scope="main_agent"),
            llm=object(),
        )
    }

    assert available_agent_tools(parent) == []


def test_delegated_agent_execution_rejects_self_delegation() -> None:
    parent = AgentState.new(
        agent_profile=AgentProfile(name="coordinator", scope="main_agent"),
        llm=object(),
    )
    parent.available_agent_states = {"coordinator": parent}
    step = PlanStep(
        id="E1",
        tool="coordinator",
        plan="Delegate this task.",
        args={"task": "Delegate this task."},
    )

    register_agent_tools([parent.agent_profile])
    result = execute_step(step, agent_state=parent)

    assert result.output.result["status"] == "rejected"
    assert "cannot delegate to itself" in result.output.result["error"]
