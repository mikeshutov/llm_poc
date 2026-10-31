from __future__ import annotations

from agent_runtime.models.main_state import MainState
from capabilities.discovery.attribute_discovery import attribute_discovery


class FakeAttributeClassifier:
    def __init__(self) -> None:
        self.context = ""
        self.attributes = []

    def relevant_attributes(self, context: str, attributes: list[str]) -> list[str]:
        self.context = context
        self.attributes = attributes
        return ["food.preference", "not-a-valid-attribute"]


def test_attribute_discovery_sends_context_and_all_available_attributes() -> None:
    state = MainState.new(task="Find a dinner recipe", agent_profiles=[])
    client = FakeAttributeClassifier()

    result = attribute_discovery(state, client=client)

    assert client.context == "Current user request:\nFind a dinner recipe"
    assert client.attributes
    assert "food.preference" in client.attributes
    assert len(client.attributes) == 30
    assert result == ["food.preference"]


def test_attribute_discovery_limits_results_to_ten() -> None:
    state = MainState.new(task="food", agent_profiles=[])

    class FakeClient:
        def relevant_attributes(self, context: str, attributes: list[str]) -> list[str]:
            return list(attributes[:12])

    result = attribute_discovery(state, client=FakeClient())

    assert len(result) == 10
