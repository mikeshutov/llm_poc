import importlib

import pytest

from agent_runtime.context import get_current_agent_state
from agent_runtime.execution.tool_executor import execute_agent_tool
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.agent_state import AgentState
from agent_runtime.state_enrichment import enrich_state
from capabilities.discovery.models import AvailableTool, CapabilityDiscoveryResult
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY, build_agent_tool


def test_initial_enrichment_discovers_against_the_request_agent(
    monkeypatch, main_state_factory, agent_profile_factory,
):
    discovery = importlib.import_module("capabilities.discovery.capability_discovery")
    helper = agent_profile_factory(name="dynamic_helper", description="Find information")
    monkeypatch.setattr(discovery, "embed_text", lambda _: [1.0])
    monkeypatch.setattr(discovery, "_discover_persisted_tools", lambda *args: [])
    monkeypatch.setattr(discovery, "_discover_agent_profiles", lambda *args: [helper])
    monkeypatch.setattr(discovery, "_rank_candidates", lambda query, embedding, candidates: [
        (1.0, candidate) for _, _, candidate in candidates
    ])
    monkeypatch.setattr("agent_runtime.state_enrichment.load_user_profile", lambda _: None)
    state = main_state_factory(task="Find a helper")
    state.discovery_query = "Previously embedded request"
    state.discovery_query_embedding = [1.0]
    seen = []

    def discover(active, **kwargs):
        assert isinstance(active, AgentState)
        assert active is state.main_agent_state
        assert active.inputs.task == state.task
        assert kwargs["allowed_tool_names"] == active.agent_profile.tool_names
        assert kwargs["query"] == state.discovery_query
        assert kwargs["query_embedding"] is state.discovery_query_embedding
        seen.append(active)
        return discovery.capability_discovery(active, **kwargs)

    enrich_state(state, attribute_lookup=lambda _: [], capability_lookup=discover)

    assert seen == [state.main_agent_state]
    assert state.discovered_capabilities.agents[0].agent == helper.name
    assert helper.name in state.agent_states
    assert state.main_agent_state.available_agent_states is state.agent_states
    assert state.main_agent_state.inputs.discovered_capabilities is state.discovered_capabilities
    assert GLOBAL_TOOL_REGISTRY.get(helper.name).name == helper.name


@pytest.mark.parametrize("strategy", list(AgentExecutionStrategy))
def test_only_react_delegates_discover_before_the_runner(
    monkeypatch, agent_state_factory, agent_profile_factory, strategy,
):
    parent = agent_state_factory()
    child = agent_state_factory(profile=agent_profile_factory(
        name="helper", execution_strategy=strategy, can_delegate_agents=False,
    ))
    child.inputs.discovered_attribute_types = ["food.preference"]
    parent.available_agent_states = {child.agent_profile.name: child}
    GLOBAL_TOOL_REGISTRY.register(build_agent_tool(child.agent_profile))
    events = []
    capabilities = CapabilityDiscoveryResult(tools=[AvailableTool(name="math_calculate")])

    def discover(active, **kwargs):
        assert get_current_agent_state() is active
        assert active.inputs.task == "Calculate a total"
        assert kwargs["include_agents"] is False
        assert kwargs["allowed_tool_names"] == active.agent_profile.tool_names
        events.append("discover")
        return capabilities

    def run(active):
        assert get_current_agent_state() is active
        assert active is not child
        assert active.inputs.discovered_attribute_types == []
        assert active.available_agent_states is None
        if strategy == AgentExecutionStrategy.REACT:
            assert active.inputs.discovered_capabilities is capabilities
        events.append("run")
        return active

    monkeypatch.setattr("agent_runtime.execution.tool_executor.capability_discovery", discover)
    monkeypatch.setattr("agents.registry.agent_registry.get", lambda _: run)

    result = execute_agent_tool(parent, "helper", {"task": "Calculate a total"})

    assert events == (["discover", "run"] if strategy == AgentExecutionStrategy.REACT else ["run"])
    assert result.tool_name == "helper"
    assert child.inputs.discovered_attribute_types == ["food.preference"]
    assert get_current_agent_state() is None
