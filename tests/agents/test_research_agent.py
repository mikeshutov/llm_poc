from agents.research.profile import RESEARCH_PROFILE
from agents.research.prompts import build_research_rules


def test_research_agent_is_limited_to_read_only_research_categories() -> None:
    assert RESEARCH_PROFILE.name == "research"
    assert RESEARCH_PROFILE.allowed_categories == {
        "web_search",
        "knowledge",
        "books",
        "language",
    }
    assert not {"file_management", "profile"}.intersection(RESEARCH_PROFILE.allowed_category_names())


def test_research_agent_is_planner_executor_evaluator() -> None:
    assert RESEARCH_PROFILE.execution_strategy.value == "planner_executor_evaluator"
    assert RESEARCH_PROFILE.max_turns == 8


def test_research_rules_use_web_search_query_text_argument() -> None:
    rules = build_research_rules()

    assert "query_text" in rules
    assert "not `query`" in rules
