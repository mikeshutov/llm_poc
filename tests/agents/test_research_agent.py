from agents.research.profile import RESEARCH_PROFILE
from agents.research.prompts import build_research_rules, load_research_template


def test_research_agent_can_research_and_write_a_workspace_report() -> None:
    assert RESEARCH_PROFILE.name == "research"
    assert RESEARCH_PROFILE.allowed_categories == {
        "web_search",
        "knowledge",
        "books",
        "language",
        "workspace",
    }
    assert not {"file_management", "profile"}.intersection(RESEARCH_PROFILE.allowed_category_names())
    assert {tool.name for tool in RESEARCH_PROFILE.static_tools} >= {
        "workspace_file_write",
        "workspace_file_read",
        "workspace_file_list",
        "workspace_file_complete",
    }


def test_research_agent_is_planner_executor_evaluator() -> None:
    assert RESEARCH_PROFILE.execution_strategy.value == "planner_executor_evaluator"
    assert RESEARCH_PROFILE.max_turns == 8


def test_research_rules_use_web_search_query_text_argument() -> None:
    rules = build_research_rules()

    assert "query_text" in rules
    assert "not `query`" in rules
    assert "research-{topic}.md" in rules


def test_research_template_is_loaded_from_agents_folder() -> None:
    template = load_research_template()

    assert template.startswith("# Research: {{topic}}")
    assert "## Evidence and sources" in template
