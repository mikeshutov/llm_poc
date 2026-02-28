from typing import Any, Dict

from websearch.models.search_type import SearchType
from websearch.models.web_search_params import WebSearchParams
from intent_layer.models.parsed_request import QueryDetails
from websearch.clients.brave_client import BraveSearchClient

#probably just use query string for now but we can have other params to illustrate what can be done
def generic_web_search(
    details: QueryDetails,
    *,
    country: str = "CA",
    count: int = 5,
    **params: Any,
) -> Dict[str, Any]:
    search_type = details.search_type if details.search_type else SearchType.WEB_SEARCH
    brave_client = BraveSearchClient.from_env()
    search_results = dict[str, Any]
    match search_type:
        case SearchType.NEWS_SEARCH:
            search_results = brave_client.news_search(details.query_text)
        case SearchType.SUGGESTION_SEARCH:
            search_results = brave_client.suggest(details.query_text)
        case _:
            search_results = brave_client.web_search(
                WebSearchParams(
                    q=details.query_text,
                    country=country,
                    count=count,
                    extra_params=params,
                )
            )

    return search_results
