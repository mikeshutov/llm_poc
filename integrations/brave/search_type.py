from enum import Enum


class SearchType(str, Enum):
    WEB_SEARCH = "web_search"
    NEWS_SEARCH = "news_search"
    SUGGESTION_SEARCH = "suggestion_search"

