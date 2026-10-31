from __future__ import annotations

import importlib
from types import SimpleNamespace

from integrations.web_fetch.client import WebFetchClient
from capabilities.adapters.web.web_fetch import web_fetch


def test_web_fetch_client_extracts_page_details(monkeypatch) -> None:
    response = SimpleNamespace(
        url="https://example.com/article",
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text="""
            <html><head><title>Example Article</title>
            <meta name="description" content="An article summary."></head>
            <body><script>ignore me</script><h1>Headline</h1><p>Readable content.</p></body></html>
        """,
    )
    response.raise_for_status = lambda: None
    monkeypatch.setattr("integrations.web_fetch.client.requests.get", lambda *args, **kwargs: response)

    page = WebFetchClient().fetch("https://example.com/article")

    assert page.title == "Example Article"
    assert page.description == "An article summary."
    assert page.text == "Example Article Headline Readable content."
    assert "ignore me" not in page.text


def test_web_fetch_returns_typed_tool_result(monkeypatch) -> None:
    web_fetch_module = importlib.import_module("capabilities.adapters.web.web_fetch")
    monkeypatch.setattr(
        web_fetch_module._client,
        "fetch",
        lambda _: SimpleNamespace(
            url="https://example.com/article",
            title="Example Article",
            description="An article summary.",
            text="Readable content.",
        ),
    )

    result = web_fetch.invoke({"url": "https://example.com/article"})

    assert result.evidence[0].tool_name == "web_fetch"
    assert result.evidence[0].entity_type == "web_page"
    assert result.evidence[0].urls[0].url == "https://example.com/article"
