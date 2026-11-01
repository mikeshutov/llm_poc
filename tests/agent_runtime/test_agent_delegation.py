from agent_runtime.execution.tool_executor import _delegated_task
from agent_runtime.models.plan import PlanStep
from agent_runtime.planning.tool_wrapper import build_tools
from agent_runtime.strategies.react.decision import _build_agent_delegation_tools
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE, DocumentWritingDelegationArgs


def test_delegated_agent_receives_explicit_task_instead_of_hypothesis() -> None:
    step = PlanStep(
        id="E1",
        agent="document_writing",
        plan="The document will be created successfully.",
        hypothesis="The document will be created successfully.",
        args={"task": "Create short_computer_joke.docx as a DOCX containing the supplied joke."},
    )

    delegated_task = _delegated_task(step)

    assert delegated_task == "Create short_computer_joke.docx as a DOCX containing the supplied joke."
    assert "The document will be created successfully" not in delegated_task


def test_agent_delegation_tool_keeps_explicit_task_schema() -> None:
    state = AgentState.new(
        agent_profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    tool = build_tools(_build_agent_delegation_tools([state]))[0]

    assert list(tool.args_schema.model_fields) == ["task", "evidence_ids"]
    assert tool.args_schema is DocumentWritingDelegationArgs
    assert tool.metadata["agent_delegation"] is True


def test_agent_delegation_tool_accepts_evidence_ids() -> None:
    state = AgentState.new(
        agent_profile=AgentProfile(name="document_writing", scope="main_agent"),
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    tool = build_tools(_build_agent_delegation_tools([state]))[0]

    parsed = tool.args_schema.model_validate({
        "task": "Create a research paper.",
        "evidence_ids": ["evidence-1", "evidence-1", "  evidence-2  "],
    })

    assert parsed.task == "Create a research paper."
    assert parsed.evidence_ids == ["evidence-1", "evidence-1", "  evidence-2  "]
