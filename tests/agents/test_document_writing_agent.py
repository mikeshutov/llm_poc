from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.strategies.react.subagent_prompt import build_subagent_decision_prompt


def test_document_writing_agent_exposes_safe_file_tools() -> None:
    assert DOCUMENT_WRITING_PROFILE.name == "document_writing"
    assert DOCUMENT_WRITING_PROFILE.tool_names == {
        "file_read",
        "file_list",
        "memory_lookup_evidence",
        "web_fetch",
        "writing_templates",
        "workspace_file_write",
        "workspace_file_read",
        "workspace_file_edit",
        "workspace_file_list",
        "workspace_file_promote",
        "update_memory_state",
    }
    assert "file_delete" not in DOCUMENT_WRITING_PROFILE.tool_names


def test_document_writing_agent_is_react() -> None:
    assert DOCUMENT_WRITING_PROFILE.execution_strategy.value == "react"


def test_document_writing_description_distinguishes_supported_formats() -> None:
    assert "TXT, Markdown (.md), DOCX, and PDF" in DOCUMENT_WRITING_PROFILE.description
    assert "Do not use it for HTML pages, CSS, JavaScript" in DOCUMENT_WRITING_PROFILE.description


def test_document_writing_react_prompt_exposes_file_creation_rules(agent_state_factory) -> None:
    state = agent_state_factory(
        profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )

    prompt, tools = build_subagent_decision_prompt(state)

    assert "workspace_file_write" in {tool.name for tool in tools}
    assert "workspace_file_promote" in {tool.name for tool in tools}
    assert "select a writing template before calling workspace_file_write" in prompt.build()
    assert "required_fields" in prompt.build()
    assert "OUTPUT CONTRACT" in prompt.build()
    assert '"status"' in prompt.build()


def test_document_writing_prompt_exposes_parent_evidence_ids(agent_state_factory) -> None:
    state = agent_state_factory(
        profile=DOCUMENT_WRITING_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.inputs.evidence_ids = ["evidence-1", "evidence-2"]

    prompt, _ = build_subagent_decision_prompt(state)

    assert prompt.sections_raw["evidence_ids"] == ["evidence-1", "evidence-2"]
