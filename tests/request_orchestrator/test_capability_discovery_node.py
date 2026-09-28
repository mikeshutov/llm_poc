from __future__ import annotations

from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.state_enrichment import enrich_state
from request_orchestrator.shared.discovery.models import CapabilityDiscoveryResult


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
