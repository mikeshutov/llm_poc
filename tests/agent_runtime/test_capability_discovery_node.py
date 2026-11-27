from __future__ import annotations

from capabilities.discovery.models import CapabilityDiscoveryResult
from agent_runtime.state_enrichment import enrich_state


def test_capability_node_stores_shared_discovery_results(main_state_factory) -> None:
    attribute_lookup = lambda _: ["food.preference"]
    capability_lookup = lambda _, **kwargs: CapabilityDiscoveryResult()
    state = main_state_factory(task="food", agent_profiles=[])

    enrich_state(
        state,
        attribute_lookup=attribute_lookup,
        capability_lookup=capability_lookup,
    )

    assert state.discovered_attribute_types == ["food.preference"]
    assert state.discovered_capabilities == CapabilityDiscoveryResult()
