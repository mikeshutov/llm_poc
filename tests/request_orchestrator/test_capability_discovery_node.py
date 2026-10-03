from __future__ import annotations

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_inputs import AgentInputs
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.discovery.models import CapabilityDiscoveryResult
from request_orchestrator.shared.state_enrichment import enrich_agent_state, enrich_state


def test_capability_node_stores_shared_discovery_results() -> None:
    attribute_lookup = lambda _: ["food.likes"]
    capability_lookup = lambda _, **kwargs: CapabilityDiscoveryResult()
    state = MainState.new(task="food", agent_profiles=[])

    enrich_state(
        state,
        attribute_lookup=attribute_lookup,
        capability_lookup=capability_lookup,
    )

    assert state.discovered_attribute_types == ["food.likes"]
    assert state.discovered_capabilities == CapabilityDiscoveryResult()


def test_agent_enrichment_preserves_discoveries_and_hydrates_profile_for_replanning() -> None:
    profile = AgentProfile(name="test_agent", scope="test")
    agent_state = AgentState(
        agent_profile=profile,
        inputs=AgentInputs.new(task="Find restaurants", request_task="Find restaurants"),
        llm=object(),
    )
    agent_state.node_states.evaluator.missing_information = ["dietary restrictions"]
    discovered_capabilities = CapabilityDiscoveryResult()
    attribute_states = []
    capability_states = []
    loaded_states = []

    def attribute_lookup(state: MainState) -> list[str]:
        attribute_states.append(state)
        return ["food.likes"]

    def capability_lookup(state: MainState, **kwargs) -> CapabilityDiscoveryResult:
        capability_states.append((state, kwargs))
        return discovered_capabilities

    def profile_loader(state: MainState) -> MainState:
        loaded_states.append(state)
        return state

    result = enrich_agent_state(
        agent_state,
        attribute_lookup=attribute_lookup,
        capability_lookup=capability_lookup,
        profile_loader=profile_loader,
    )

    assert result.inputs.discovered_attribute_types == ["food.likes"]
    assert result.inputs.discovered_capabilities is discovered_capabilities
    assert "dietary restrictions" in attribute_states[0].task
    assert "dietary restrictions" in capability_states[0][0].task
    assert capability_states[0][1] == {"include_agents": False}
    assert loaded_states[0].discovered_attribute_types == ["food.likes"]
