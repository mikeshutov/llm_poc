from agents.research.profile import RESEARCH_PROFILE


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
