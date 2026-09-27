from __future__ import annotations

import json

from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.request_analysis.attribute_discovery import attribute_discovery


class FakeAttributeClassifier:
    def __init__(self) -> None:
        self.context = ""
        self.attributes = []

    def relevant_attributes(self, context: str, attributes: list[str]) -> list[str]:
        self.context = context
        self.attributes = attributes
        return ["food.likes", "not-a-valid-attribute"]


def test_attribute_discovery_sends_context_and_all_available_attributes() -> None:
    state = MainState.new(task="Find a dinner recipe", agent_profiles=[])
    client = FakeAttributeClassifier()

    result = attribute_discovery(state, client=client)

    context = json.loads(client.context)
    assert context["task"] == "Find a dinner recipe"
    assert client.attributes
    assert "food.likes" in client.attributes
    assert len(client.attributes) == 36
    assert result == ["food.likes"]
