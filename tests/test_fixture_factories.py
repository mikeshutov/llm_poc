from uuid import uuid4

import pytest
from requests import HTTPError

from agent_runtime.models.plan import Plan, PlanStep
from agent_runtime.models.agent_profile import AgentKind
from capabilities.adapters.writing.templates import load_writing_template_content, load_writing_templates


def test_state_factory_supports_minimal_and_configured_states(agent_profile_factory, agent_state_factory):
    minimal = agent_state_factory()
    profile = agent_profile_factory(name="custom", max_turns=2)
    llm = object()
    configured = agent_state_factory(
        profile=profile, profile_overrides={"max_turns": 5}, task="Investigate", llm=llm,
    )

    assert minimal.agent_profile.name == "test_agent"
    assert minimal.inputs.task == ""
    assert configured.agent_profile.name == "custom"
    assert configured.agent_profile.max_turns == 5
    assert profile.max_turns == 2
    assert configured.inputs.task == "Investigate"
    assert configured.llm is llm
    assert minimal.inputs is not configured.inputs


def test_state_factory_infers_tools_from_plan(agent_state_factory):
    plan = Plan(steps=[PlanStep(id="search", tool="web_search", plan="Search", args={})])
    state = agent_state_factory(plan=plan)

    assert state.node_states.planner.plan is plan
    assert state.node_states.planner.plan_count == 1
    assert [tool.name for tool in state.agent_profile.static_tools] == ["web_search"]


def test_user_agent_profile_factory_supports_defaults_and_overrides(user_agent_profile_factory):
    default = user_agent_profile_factory()
    customized = user_agent_profile_factory(name="helper", description="Research helper", max_turns=2)

    assert default.kind == customized.kind == AgentKind.USER_AGENT
    assert default.name == "assistant"
    assert customized.name == "helper"
    assert customized.description == "Research helper"
    assert customized.max_turns == 2
    customized.static_tools.append(object())
    assert default.static_tools == []


def test_state_factory_sets_user_without_mutating_supplied_context(agent_state_factory):
    alice = agent_state_factory(user_id="alice")
    bob = agent_state_factory(user_id="bob", execution_context=alice.execution_context)

    assert alice.execution_context.user_profile.user_id == "alice"
    assert bob.execution_context.user_profile.user_id == "bob"
    assert bob.execution_context is not alice.execution_context
    assert bob.execution_context.user_profile is not alice.execution_context.user_profile


def test_execution_context_factory_supports_ids_and_fresh_defaults(execution_context_factory):
    first = execution_context_factory(user_id="alice", conversation_id="conversation-1", roundtrip_id=uuid4())
    second = execution_context_factory()

    assert first.user_profile.user_id == "alice"
    assert first.conversation_id == "conversation-1"
    assert first.roundtrip_id is not None
    assert second.conversation_id is None
    assert second.roundtrip_id is None
    assert first.user_profile is not second.user_profile
    assert first.conversation_context is not second.conversation_context
    overridden = execution_context_factory(user_id="bob", user_profile=first.user_profile)
    assert overridden.user_profile.user_id == "bob"
    assert first.user_profile.user_id == "alice"


def test_memory_factory_builds_independent_small_and_large_states(working_memory_state_factory):
    small = working_memory_state_factory(task="Small")
    large_content = "Checked finding\n" * 1000
    large = working_memory_state_factory(content=large_content, file_name="memory/custom.md")

    assert "Small" in small.read_working_memory()
    assert large.read_working_memory() == large_content.strip()
    assert small.working_memory_file_name != large.working_memory_file_name


def test_repository_factory_copies_seed_records(tool_repository_factory):
    record_id = uuid4()
    records = {record_id: {"example": True}}
    seeded = tool_repository_factory(records=records)
    empty = tool_repository_factory()
    seeded.records.clear()

    assert record_id in records
    assert empty.records == {}
    assert seeded.compacted_ids is not empty.compacted_ids


def test_plans_have_fresh_defaults_and_composable_overrides(default_plan, plan_factory, plan_step_factory):
    other = plan_factory()
    assert default_plan.steps[0].id == "E1"
    assert default_plan.steps[0].tool == "web_search"
    assert default_plan.steps[0].db_id != other.steps[0].db_id
    default_plan.steps[0].args["query"] = "changed"
    assert other.steps[0].args == {}

    args = {"filters": ["news"]}
    first = plan_step_factory(id="first", tool="web_search", args=args)
    second = plan_step_factory(id="second", tool="web_fetch", args={"url": "https://example.com"})
    plan = plan_factory(steps=[first, second])
    plan.steps[0].args["filters"].append("books")

    assert [step.step_index for step in plan.steps] == [0, 1]
    assert first.args == args == {"filters": ["news"]}
    assert second.step_index == 0
    assert plan_factory(steps=[]).steps == []


def test_conversation_factory_copies_mutable_overrides(conversation_message_factory):
    payload = {"items": [1]}
    first = conversation_message_factory(payload=payload)
    second = conversation_message_factory(payload=payload)
    first.payload["items"].append(2)

    assert second.payload == payload == {"items": [1]}
    assert first.id != second.id


def test_llm_factory_copies_tool_calls_and_records_prompts(llm_factory):
    calls = [{"name": "lookup", "args": {"query": "test"}}]
    llm = llm_factory("answer", tool_calls=calls, input_tokens=10, output_tokens=2)
    response = llm.invoke("question")
    response.tool_calls[0]["args"]["query"] = "changed"

    assert llm.prompts == ["question"]
    assert response.usage_metadata["total_tokens"] == 12
    assert llm.tool_calls == calls


def test_topic_factories_provide_independent_data(product_factory, user_attribute_factory, evidence_factory):
    first_product, second_product = product_factory(), product_factory(name="Other")
    assert first_product.name == "Test product"
    assert second_product.name == "Other"
    first_attribute, second_attribute = user_attribute_factory(), user_attribute_factory()
    first_attribute.value.append("coffee")
    assert second_attribute.value == ["pizza"]
    data = {"items": [1]}
    evidence = evidence_factory(data=data)
    evidence.data["items"].append(2)
    assert data == {"items": [1]}


def test_http_factory_provides_json_and_failure_responses(http_response_factory):
    response = http_response_factory(payload={"ok": True})
    assert response.json() == {"ok": True}
    response.raise_for_status()
    with pytest.raises(HTTPError):
        http_response_factory(status_code=503).raise_for_status()


def test_writing_catalog_factory_has_a_usable_default(writing_catalog_factory):
    directory = writing_catalog_factory()

    assert [template.id for template in load_writing_templates(directory)] == ["essay"]
    assert load_writing_template_content("essay", directory) == "{{ body }}"
