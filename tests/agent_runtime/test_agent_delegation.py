from agent_runtime.models.plan import PlanStep
from agent_runtime.planning.tool_wrapper import build_tools
from capabilities.registry.global_registry import (
    GLOBAL_TOOL_REGISTRY, available_agent_tools, register_agent_tools,
)
from agent_runtime.execution.tool_executor import run_executor
from agent_runtime.models.plan import Plan
from agent_runtime.models.agent_profile import AgentProfile
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE, DocumentWritingDelegationArgs
from capabilities.discovery.models import AvailableAgent, CapabilityDiscoveryResult


def test_agent_tool_receives_explicit_task_and_normalizes_evidence(monkeypatch, agent_state_factory) -> None:
    parent = agent_state_factory(profile=AgentProfile(name="parent", scope="main_agent"))
    child = agent_state_factory(profile=DOCUMENT_WRITING_PROFILE)
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
    parent.node_states.planner.plan = Plan(steps=[step])
    run_executor(parent)
    result = parent.direct_tool_results[0]
    assert seen[0].inputs.task == "Create a document."
    assert seen[0].inputs.evidence_ids == ["ev-1", "ev-2"]
    assert child.inputs.task == ""
    assert not result.result.get("error")


def test_agent_delegation_tool_keeps_explicit_task_schema() -> None:
    tool = build_tools([GLOBAL_TOOL_REGISTRY.get(DOCUMENT_WRITING_PROFILE.name)])[0]

    assert list(tool.args_schema.model_fields) == ["parameters", "hypothesis", "evaluation_fields"]
    assert tool.args_schema.model_fields["parameters"].annotation is DocumentWritingDelegationArgs


def test_agent_delegation_tool_accepts_evidence_ids() -> None:
    tool = build_tools([GLOBAL_TOOL_REGISTRY.get(DOCUMENT_WRITING_PROFILE.name)])[0]

    parsed = tool.args_schema.model_fields["parameters"].annotation.model_validate({
        "task": "Create a research paper.",
        "evidence_ids": ["evidence-1", "evidence-1", "  evidence-2  "],
    })

    assert parsed.task == "Create a research paper."
    assert parsed.evidence_ids == ["evidence-1", "evidence-1", "  evidence-2  "]


def test_delegated_agent_states_honor_delegation_policy(agent_state_factory) -> None:
    parent = agent_state_factory(
        profile=AgentProfile(
            name="coordinator",
            scope="main_agent",
            default_available_agent_names={"default_agent"},
        ),
    )
    default_agent = agent_state_factory(
        profile=AgentProfile(name="default_agent", scope="main_agent"),
    )
    undiscovered_agent = agent_state_factory(
        profile=AgentProfile(name="undiscovered_agent", scope="main_agent"),
    )
    blocked_agent = agent_state_factory(
        profile=AgentProfile(name="blocked_agent", scope="main_agent", delegatable=False),
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


def test_non_delegating_agent_exposes_no_delegation_targets(agent_state_factory) -> None:
    parent = agent_state_factory(
        profile=AgentProfile(
            name="leaf",
            scope="main_agent",
            can_delegate_agents=False,
        ),
    )
    parent.available_agent_states = {
        "other": agent_state_factory(
            profile=AgentProfile(name="other", scope="main_agent"),
        )
    }

    assert available_agent_tools(parent) == []


def test_delegated_agent_execution_rejects_self_delegation(agent_state_factory) -> None:
    parent = agent_state_factory(
        profile=AgentProfile(name="coordinator", scope="main_agent"),
    )
    parent.available_agent_states = {"coordinator": parent}
    step = PlanStep(
        id="E1",
        tool="coordinator",
        plan="Delegate this task.",
        args={"task": "Delegate this task."},
    )

    register_agent_tools([parent.agent_profile])
    parent.node_states.planner.plan = Plan(steps=[step])
    run_executor(parent)
    result = parent.direct_tool_results[0]

    assert result.result["status"] == "rejected"
    assert "cannot delegate to itself" in result.result["error"]
