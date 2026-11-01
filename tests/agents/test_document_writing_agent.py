from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.strategies.react.prompt import build_react_decision_prompt


def test_document_writing_agent_exposes_safe_file_tools() -> None:
    assert DOCUMENT_WRITING_PROFILE.name == "document_writing"
    assert DOCUMENT_WRITING_PROFILE.tool_names == {
        "file_create",
        "file_read",
        "file_edit",
        "file_list",
        "file_move",
        "memory_lookup_evidence",
        "web_fetch",
        "writing_templates",
    }
    assert "file_delete" not in DOCUMENT_WRITING_PROFILE.tool_names


def test_document_writing_agent_is_react() -> None:
    assert DOCUMENT_WRITING_PROFILE.execution_strategy.value == "react"


def test_document_writing_react_prompt_exposes_file_creation_rules() -> None:
    state = AgentState.new(
        agent_profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, tools = build_react_decision_prompt(state)

    assert "file_create" in {tool.name for tool in tools}
    assert "file_create for new documents" in prompt.build()


def test_document_writing_prompt_exposes_parent_evidence_ids() -> None:
    state = AgentState.new(
        agent_profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.inputs.evidence_ids = ["evidence-1", "evidence-2"]

    prompt, _ = build_react_decision_prompt(state)

    assert prompt.sections_raw["evidence_ids"] == ["evidence-1", "evidence-2"]
