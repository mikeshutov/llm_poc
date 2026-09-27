import sys
from types import ModuleType, SimpleNamespace

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from common.html_text import html_to_plain_text
from integrations.brave.models import NewsResult, NewsSearchQuery, NewsSearchResponse, WebSearchResponse, WebSearchResult
from products.models.product_result import ProductResult
from products.models.product_search_results import ProductSearchResults
from products.models.product_source import ProductSource
from request_orchestrator.models.evidence import EvidenceView
from request_orchestrator.shared.tool_adapter.products.find_products_web import _product_summary
from request_orchestrator.shared.tool_adapter.products.find_products_web import _tool_result as web_product_tool_result
from request_orchestrator.shared.tool_adapter.search.brave_news_search import _tool_result as brave_news_tool_result
from request_orchestrator.shared.tool_adapter.search.generic_web_search import _news_search_tool_result, _web_search_tool_result
from reranker.models import RerankerCandidate


def test_html_to_plain_text_normalizes_entities_and_whitespace() -> None:
    assert html_to_plain_text("DEWALT&#x20;20V&nbsp;&amp;&#32;XR") == "DEWALT 20V & XR"


def test_evidence_view_normalizes_llm_facing_text() -> None:
    evidence = EvidenceView(
        title="<b>DEWALT</b>&#x20;20V",
        summary="<p>Brushless&nbsp;drill.</p>",
        llm_metadata={"seller": "<em>Amazon</em>&#x20;Store"},
    )

    compact_view = evidence.compact_view()

    assert compact_view["title"] == "DEWALT 20V"
    assert compact_view["summary"] == "Brushless drill."
    assert compact_view["metadata"] == {"seller": "Amazon Store"}


def test_evidence_view_projects_semantic_fields_to_reranker_candidate() -> None:
    evidence = EvidenceView(
        item_id="drill-1",
        title="DEWALT 20V",
        summary="Brushless drill.",
        urls=[{"url": "https://example.com/drill"}],
        image_url="https://example.com/drill.jpg",
        source="product-search",
        llm_metadata={"category": "Tools", "color": "Yellow"},
    )

    candidate = evidence.to_candidate()

    assert candidate.id == "drill-1"
    assert candidate.fields == {
        "title": "DEWALT 20V",
        "summary": "Brushless drill.",
        "category": "Tools",
        "color": "Yellow",
    }


def test_reranker_candidate_contains_only_standardized_text() -> None:
    candidate = RerankerCandidate(
        id="drill-1",
        fields={"title": "DEWALT 20V", "description": "Brushless drill.", "seller": "Amazon Store"},
    )

    assert candidate.id == "drill-1"
    assert candidate.fields == {"title": "DEWALT 20V", "description": "Brushless drill.", "seller": "Amazon Store"}


def test_web_search_evidence_strips_html_from_brave_summary() -> None:
    result = _web_search_tool_result(
        WebSearchResponse(
            query="Toronto weather",
            results=[
                WebSearchResult(
                    title="Weather",
                    url="https://example.com/weather",
                    description="<strong>Toronto</strong> weather &amp; <em>wind</em>.",
                )
            ],
        )
    )

    assert result.evidence[0].summary == "Toronto weather & wind."


def test_news_search_evidence_strips_html_from_brave_summary() -> None:
    result = _news_search_tool_result(
        NewsSearchResponse(
            query=NewsSearchQuery(original="Toronto weather"),
            results=[
                NewsResult(
                    title="Weather",
                    url="https://example.com/weather",
                    description="Forecast for <b>Toronto</b><br>today.",
                )
            ],
        )
    )

    assert result.evidence[0].summary == "Forecast for Toronto today."


def test_standalone_brave_news_evidence_strips_html_from_summary() -> None:
    result = brave_news_tool_result(
        NewsSearchResponse(
            query=NewsSearchQuery(original="Toronto weather"),
            results=[
                NewsResult(
                    title="Weather",
                    url="https://example.com/weather",
                    description="Forecast for <b>Toronto</b><br>today.",
                )
            ],
        )
    )

    assert result.evidence[0].summary == "Forecast for Toronto today."


def test_brave_shopping_product_evidence_strips_html_from_summary() -> None:
    product = ProductResult(
        id="product-1",
        name="Weather Radio",
        description="<b>Weather</b> radio &amp; alert receiver.",
        category=None,
        color=None,
        style=None,
        gender=None,
        season=None,
        year=None,
        price=None,
        source=ProductSource.WEB,
    )

    assert _product_summary(product) == "Weather radio & alert receiver."


def test_product_result_metadata_is_not_repeated_on_each_evidence_item() -> None:
    product = ProductResult(
        id="product-1",
        name="Weather Radio",
        description="Alert receiver.",
        category=None,
        color=None,
        style=None,
        gender=None,
        season=None,
        year=None,
        price=None,
        source=ProductSource.WEB,
    )

    result = web_product_tool_result(
        ProductSearchResults(
            internal_results=[],
            external_results=[product],
            retrieved_count=12,
            reranked=True,
        )
    )

    assert result.tool_metadata.model_dump(exclude_none=True) == {
        "retrieved_count": 12,
        "reranked": True,
        "product_source": "web",
    }
    assert result.evidence[0].llm_metadata == {}
