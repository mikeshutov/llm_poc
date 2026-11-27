from __future__ import annotations

from serviceclients.attribute_classifier.client import AttributeClassifierClient


def test_client_returns_relevant_attributes(http_response_factory, monkeypatch) -> None:
    calls = {}

    def post(url, *, json, timeout):
        calls.update(url=url, json=json, timeout=timeout)
        return http_response_factory(payload={"relevant_attributes": ["waterproof rating"]})

    monkeypatch.setattr("serviceclients.attribute_classifier.client.requests.post", post)

    result = AttributeClassifierClient(base_url="http://classifier").relevant_attributes(
        "winter hiking", ["waterproof rating", "color"]
    )

    assert result == ["waterproof rating"]
    assert calls["url"] == "http://classifier/classify"
    assert calls["json"] == {
        "context": "winter hiking",
        "attributes": ["waterproof rating", "color"],
    }
