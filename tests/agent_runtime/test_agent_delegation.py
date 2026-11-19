from agent_runtime.execution.tool_executor import _delegated_task
from agent_runtime.execution.tool_executor import _agent_completion_result
from agent_runtime.execution.tool_executor import _execute_top_level_step
from agent_runtime.execution.tool_executor import _profile_for_delegated_run
from agent_runtime.models.plan import PlanStep
from agent_runtime.planning.tool_wrapper import build_tools
from agent_runtime.planning.agent_delegation import delegated_agent_states
from agent_runtime.strategies.react.decision import _build_agent_delegation_tools
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_profile import AgentDelegationRole
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.runner import AgentRunner
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE, DocumentWritingDelegationArgs
from agents.research.profile import RESEARCH_PROFILE
from agents.delegate.profile import DELEGATE_PROFILE
from agent_runtime.working_memory import initialize_working_memory_file
from capabilities.discovery.discover_capabilities_tool import CAPABILITY_DISCOVERY_TOOL
from retrieval.evidence import EvidenceView, ToolResult


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


def test_research_coordinator_delegates_to_a_non_delegating_subagent() -> None:
    child_profile = _profile_for_delegated_run(RESEARCH_PROFILE, DELEGATE_PROFILE)
    child_state = AgentState.new(
        agent_profile=child_profile,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    child_state.available_agent_states = {DELEGATE_PROFILE.name: child_state}

    assert RESEARCH_PROFILE.can_delegate_agents is True
    assert RESEARCH_PROFILE.delegation_role == AgentDelegationRole.COORDINATOR
    assert child_profile.can_delegate_agents is False
    assert child_profile.delegation_role == AgentDelegationRole.LEAF
    assert child_profile.name == DELEGATE_PROFILE.name
    assert delegated_agent_states(child_state) == []


def test_delegate_is_a_default_research_capability() -> None:
    research_state = AgentState.new(
        agent_profile=RESEARCH_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    delegate_state = AgentState.new(
        agent_profile=DELEGATE_PROFILE,
        task="Research a subquestion",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    research_state.available_agent_states = {DELEGATE_PROFILE.name: delegate_state}
    research_state.inputs.discovered_capabilities = None

    visible_agents = delegated_agent_states(research_state)

    assert RESEARCH_PROFILE.default_available_agent_names == {"delegate"}
    assert [state.agent_profile.name for state in visible_agents] == ["delegate"]
    assert DELEGATE_PROFILE.execution_strategy.value == "react"
    assert not DELEGATE_PROFILE.required_completion_tools
    assert "# Working Memory" in DELEGATE_PROFILE.working_memory_template
    assert "Research a subquestion" in delegate_state.render_working_memory()
    assert {
        "discover_capabilities",
        "workspace_file_write",
        "workspace_file_read",
        "workspace_file_edit",
        "workspace_file_list",
    } == DELEGATE_PROFILE.tool_names
    assert "search_products" not in DELEGATE_PROFILE.tool_names
    assert "code_execute" not in DELEGATE_PROFILE.tool_names
    assert "file_create" not in DELEGATE_PROFILE.tool_names
    assert CAPABILITY_DISCOVERY_TOOL.metadata["compactable"] is False


def test_runtime_creates_delegate_working_memory_file_before_execution(tmp_path, monkeypatch) -> None:
    execution_context = AgentExecutionContext.new(conversation_id="conversation-1")
    state = AgentState.new(
        agent_profile=DELEGATE_PROFILE,
        task="Investigate context engineering techniques.",
        execution_context=execution_context,
        llm=object(),
    )
    monkeypatch.setattr(
        "agent_runtime.working_memory.conversation_workspace_path",
        lambda conversation_id, user_id=None: tmp_path,
    )

    initialize_working_memory_file(state)

    assert state.working_memory_file_name is not None
    memory_path = tmp_path / state.working_memory_file_name
    assert memory_path.is_file()
    content = memory_path.read_text(encoding="utf-8")
    assert "Investigate context engineering techniques." in content
    assert state.working_memory_file_name in content
    assert "{memory_path}" not in content


def test_delegate_reinjects_modified_runtime_working_memory(tmp_path, monkeypatch) -> None:
    execution_context = AgentExecutionContext.new(conversation_id="conversation-1")
    state = AgentState.new(
        agent_profile=DELEGATE_PROFILE,
        task="Investigate context engineering techniques.",
        execution_context=execution_context,
        llm=object(),
    )
    monkeypatch.setattr(
        "agent_runtime.working_memory.conversation_workspace_path",
        lambda conversation_id, user_id=None: tmp_path,
    )
    monkeypatch.setattr(
        "assets.files.storage.conversation_workspace_path",
        lambda conversation_id, user_id=None: tmp_path,
    )

    initialize_working_memory_file(state)
    memory_path = tmp_path / state.working_memory_file_name
    memory_path.write_text(
        "# Working Memory\n\n## Current State\nSources compared.",
        encoding="utf-8",
    )

    rendered = state.render_working_memory()

    assert "Sources compared." in rendered
    assert "Investigate context engineering techniques." not in rendered
    assert state.working_memory_file_name in rendered


def test_research_cannot_delegate_to_itself() -> None:
    research_state = AgentState.new(
        agent_profile=RESEARCH_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    research_state.available_agent_states = {
        RESEARCH_PROFILE.name: AgentState.new(
            agent_profile=RESEARCH_PROFILE,
            execution_context=AgentExecutionContext.new(),
            llm=object(),
        ),
        DELEGATE_PROFILE.name: AgentState.new(
            agent_profile=DELEGATE_PROFILE,
            execution_context=AgentExecutionContext.new(),
            llm=object(),
        ),
    }
    research_state.inputs.discovered_capabilities = None

    assert [state.agent_profile.name for state in delegated_agent_states(research_state)] == ["delegate"]


def test_execution_rejects_self_delegation_even_if_plan_bypasses_visibility() -> None:
    state = AgentState.new(
        agent_profile=RESEARCH_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    step = PlanStep(
        id="E-self",
        plan="delegate to research",
        agent=RESEARCH_PROFILE.name,
        args={"task": "Repeat the parent research task."},
    )

    result = _execute_top_level_step(state, step, resolved_args=step.args)

    assert result.output.result["status"] == "rejected"
    assert "cannot delegate to itself" in result.output.result["error"]


def test_delegate_completion_returns_runtime_evidence_without_file_handoff() -> None:
    state = AgentState.new(
        agent_profile=DELEGATE_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    state.direct_tool_results = [
        ToolResult(
            tool_name="web_search",
            evidence=[EvidenceView(source="web_search", summary="intermediate source")],
        ),
    ]

    result = _agent_completion_result(
        agent_name="delegate",
        state=state,
        plan_step_id=None,
        task="Compare historical price series.",
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].source == "web_search"
    assert "runtime working memory" in result.result["summary"]


def test_delegate_completion_excludes_private_working_memory_evidence() -> None:
    state = AgentState.new(
        agent_profile=DELEGATE_PROFILE,
        llm=object(),
    )
    state.working_memory_file_name = ".delegate-working-memory-abc.md"
    state.direct_tool_results = [
        ToolResult(
            tool_name="workspace_file_write",
            evidence=[EvidenceView(
                source="written",
                metadata={"path": state.working_memory_file_name},
                data={"title": state.working_memory_file_name},
            )],
        ),
        ToolResult(
            tool_name="web_search",
            evidence=[EvidenceView(source="web_search", summary="research evidence")],
        ),
    ]

    result = _agent_completion_result(
        agent_name="delegate",
        state=state,
        plan_step_id=None,
        task="Research one subquestion.",
    )

    assert [item.summary for item in result.evidence] == ["research evidence"]


def test_non_research_delegation_preserves_child_permission() -> None:
    child_profile = _profile_for_delegated_run(DOCUMENT_WRITING_PROFILE, RESEARCH_PROFILE)

    assert child_profile is RESEARCH_PROFILE


def test_runner_applies_its_configured_profile() -> None:
    state = AgentState.new(
        agent_profile=RESEARCH_PROFILE,
        execution_context=AgentExecutionContext.new(),
        llm=object(),
    )
    class IdentityStrategy:
        def run(self, agent_state, *, thread_id):
            return agent_state

    runner = AgentRunner(DELEGATE_PROFILE, strategy=IdentityStrategy())

    result = runner.run(state)

    assert result.agent_profile is DELEGATE_PROFILE
    assert result.agent_profile.can_delegate_agents is False
