from __future__ import annotations

from capabilities.discovery.attribute_discovery import attribute_discovery


def test_attribute_discovery_sends_context_and_all_available_attributes(
    attribute_classifier_factory,
    main_state_factory,
) -> None:
    state = main_state_factory(task="Find a dinner recipe", agent_profiles=[])
    client = attribute_classifier_factory(result=["food.preference", "not-a-valid-attribute"])

    result = attribute_discovery(state, client=client)

    assert client.context == "Current user request:\nFind a dinner recipe"
    assert client.attributes
    assert "food.preference" in client.attributes
    assert len(client.attributes) == 30
    assert result == ["food.preference"]


def test_attribute_discovery_limits_results_to_ten(attribute_classifier_factory, main_state_factory) -> None:
    state = main_state_factory(task="food", agent_profiles=[])

    client = attribute_classifier_factory(select=lambda attributes: attributes[:12])
    result = attribute_discovery(state, client=client)

    assert len(result) == 10
