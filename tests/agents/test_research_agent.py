from agent_runtime.planning.prompts.planner_prompt import build_planner_prompt
from agents.research.profile import RESEARCH_PROFILE
from capabilities.registry.constants import (
    TOOL_CATEGORY_BOOKS,
    TOOL_CATEGORY_FILE_MANAGEMENT,
    TOOL_CATEGORY_KNOWLEDGE,
    TOOL_CATEGORY_LANGUAGE,
    TOOL_CATEGORY_WEB_SEARCH,
)
from agents.research.prompts import build_research_rules


def test_research_agent_records_a_structured_working_memory_handoff() -> None:
    assert RESEARCH_PROFILE.name == "research"
    assert RESEARCH_PROFILE.allowed_categories == {
        TOOL_CATEGORY_WEB_SEARCH,
        TOOL_CATEGORY_KNOWLEDGE,
        TOOL_CATEGORY_BOOKS,
        TOOL_CATEGORY_LANGUAGE,
    }
    assert not {TOOL_CATEGORY_FILE_MANAGEMENT, "profile"}.intersection(RESEARCH_PROFILE.allowed_category_names())
    assert {tool.name for tool in RESEARCH_PROFILE.static_tools} >= {
        "update_memory_state",
    }
    assert not {"workspace_file_write", "workspace_file_read", "workspace_file_list"}.intersection(
        RESEARCH_PROFILE.tool_names
    )


def test_research_agent_is_planner_executor_evaluator() -> None:
    assert RESEARCH_PROFILE.execution_strategy.value == "planner_executor_evaluator"
    assert RESEARCH_PROFILE.max_turns == 10
    assert RESEARCH_PROFILE.can_delegate_agents
    assert RESEARCH_PROFILE.default_available_agent_names == {"delegate"}
    assert RESEARCH_PROFILE.working_memory_template
    assert "## Plan" in RESEARCH_PROFILE.working_memory_template
    assert "## Evaluation" in RESEARCH_PROFILE.working_memory_template


def test_research_rules_use_web_search_query_text_argument() -> None:
    rules = build_research_rules()

    assert "query_text" in rules
    assert "not `query`" in rules
    assert "working memory" in rules
    assert "research-{topic}.md" not in rules
    assert "document-writing agent" in rules


def test_research_uses_strategy_memory_instructions() -> None:
    template = RESEARCH_PROFILE.working_memory_template
    normalized_template = " ".join(template.split())

    assert "{memory_path}" not in template
    assert "At the start of every reasoning cycle" not in template
    assert "current roundtrip" not in normalized_template
    assert "source_tool_call_ids" not in normalized_template
    assert "discard_tool_call_ids" not in normalized_template
    assert "must not be passed in `source_tool_call_ids`" not in normalized_template
    assert "Pass the evidence IDs summarized" not in template


def test_research_delegation_guidance_is_optional_and_profile_driven(agent_state_factory) -> None:
    state = agent_state_factory(profile=RESEARCH_PROFILE, task="Research a topic")
    state.available_agent_states = {}

    prompt, _ = build_planner_prompt(state)

    assert prompt.instruction == RESEARCH_PROFILE.planner_instruction
    assert "Delegate independent research questions when useful." in prompt.instruction
    assert "avoid duplicating completed work" in prompt.instruction
    assert "parallel" not in prompt.instruction
