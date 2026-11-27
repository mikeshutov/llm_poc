from __future__ import annotations

from types import SimpleNamespace


from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentProfile
from capabilities.registry.constants import (
    TOOL_CATEGORY_PRODUCTS,
)
from agent_runtime.strategies.working_memory import load_strategy_working_memory_template
from capabilities.registry.tools import WORKING_MEMORY_TOOLS
from capabilities.registry.tools import TOOL_CATEGORIES


def test_agent_profile_caches_resolved_tools_from_constructor_inputs() -> None:
    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        allowed_categories={TOOL_CATEGORY_PRODUCTS},
        static_tools=[
            SimpleNamespace(name="custom_tool"),
        ],
    )

    expected_category_tool_names = {
        tool.name
        for tool in TOOL_CATEGORIES[TOOL_CATEGORY_PRODUCTS].tools
    }

    assert expected_category_tool_names.issubset(profile.tool_names)
    assert "custom_tool" in profile.tool_names
    assert {tool.name for tool in profile.tools} == profile.tool_names


def test_agent_profile_static_tools_override_category_tools_by_name() -> None:
    category_tool = TOOL_CATEGORIES[TOOL_CATEGORY_PRODUCTS].tools[0]
    replacement_tool = SimpleNamespace(name=category_tool.name, replacement=True)

    profile = AgentProfile(
        name="test_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        allowed_categories={TOOL_CATEGORY_PRODUCTS},
        static_tools=[replacement_tool],
    )

    resolved_tool = next(tool for tool in profile.tools if tool.name == category_tool.name)
    assert resolved_tool is replacement_tool


def test_each_strategy_receives_its_default_working_memory_and_updater() -> None:
    templates = {}
    for strategy in AgentExecutionStrategy:
        profile = AgentProfile(
            name=f"{strategy.value}_agent",
            scope=MAIN_AGENT_MODEL_SCOPE,
            execution_strategy=strategy,
            static_tools=list(WORKING_MEMORY_TOOLS),
            working_memory_template=load_strategy_working_memory_template(strategy),
        )
        templates[strategy] = profile.working_memory_template
        normalized_template = " ".join(profile.working_memory_template.split())
        assert "# Working Memory" in profile.working_memory_template
        assert "## Objective" in profile.working_memory_template
        assert "{{ objective }}" in profile.working_memory_template
        assert "update_memory_state" in profile.tool_names
        assert "working-memory update" not in normalized_template
        assert "current roundtrip" not in normalized_template
        assert "source_tool_call_ids" not in normalized_template
        assert "discard_tool_call_ids" not in normalized_template
        assert "must not be passed in `source_tool_call_ids`" not in normalized_template
        assert "Pass the evidence IDs summarized" not in profile.working_memory_template

    assert len(set(templates.values())) == len(templates)


def test_custom_working_memory_template_overrides_react_default() -> None:
    custom_template = "# Custom Memory\n\n## Checklist\n- [ ] Start"
    profile = AgentProfile(
        name="custom_react_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        execution_strategy=AgentExecutionStrategy.REACT,
        working_memory_template=custom_template,
    )

    assert profile.working_memory_template == custom_template


def test_custom_working_memory_template_overrides_strategy_default() -> None:
    custom_template = "# Custom Memory\n\n## Checklist\n- [ ] Start"
    profile = AgentProfile(
        name="custom_planner_agent",
        scope=MAIN_AGENT_MODEL_SCOPE,
        execution_strategy=AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        working_memory_template=custom_template,
    )

    assert profile.working_memory_template == custom_template
