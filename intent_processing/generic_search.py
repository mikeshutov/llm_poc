from typing import Any, Dict

from models.search_type import SearchType
from websearch.clients.brave_client import BraveSearchClient

#probably just use query string for now but we can have other params to illustrate what can be done
def generic_web_search(q: str, search_type: SearchType,country: str = "CA", count: int = 5, **params: Any) -> Dict[str, Any]:
    brave_client = BraveSearchClient.from_env()
    search_results =  dict[str, Any]
    match search_type:
        case SearchType.NEWS_SEARCH:
            search_results = brave_client.news_search(q)
        case SearchType.SUGGESTION_SEARCH:
            search_results = brave_client.suggest(q)
        case _:
            search_results = brave_client.web_search(q, search_type, country=country, count=count, **params)

    print(search_results)
    return search_results