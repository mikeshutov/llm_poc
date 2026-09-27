from __future__ import annotations

from attribute_classifier.client import AttributeClassifierClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, list[str]]:
        return {"relevant_attributes": ["waterproof rating"]}


def test_client_returns_relevant_attributes(monkeypatch, caplog) -> None:
    calls = {}

    def post(url, *, json, timeout):
        calls.update(url=url, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("attribute_classifier.client.requests.post", post)

    with caplog.at_level("INFO", logger="attribute_classifier.client"):
        result = AttributeClassifierClient(base_url="http://classifier").relevant_attributes(
            "winter hiking", ["waterproof rating", "color"]
        )

    assert result == ["waterproof rating"]
    assert calls["url"] == "http://classifier/classify"
    assert calls["json"] == {
        "context": "winter hiking",
        "attributes": ["waterproof rating", "color"],
    }
    assert "Sending attribute classification request" in caplog.text
    assert "Received attribute classification result" in caplog.text
