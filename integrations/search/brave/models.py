from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, model_validator


class WebSearchResult(BaseModel):
    title: str = ""
    url: Optional[str] = None
    description: str = ""
    image_url: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def first_str(*keys: str) -> str:
            for k in keys:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def first_url(*keys: str) -> Optional[str]:
            for k in keys:
                v = data.get(k)
                if isinstance(v, dict):
                    u = v.get("url") or v.get("original") or v.get("src")
                    if isinstance(u, str) and u.strip():
                        return u.strip()
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None

        return {
            **data,
            "title": first_str("title", "name"),
            "url": first_url("url", "link"),
            "description": first_str("description", "snippet", "summary"),
            "image_url": first_url("image_url", "image", "thumbnail", "thumbnailUrl"),
        }


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult] = []

    @model_validator(mode="before")
    @classmethod
    def _flatten_web(cls, data: Any) -> Any:
        if isinstance(data, dict) and "results" not in data:
            web = data.get("web")
            if isinstance(web, dict):
                data = {**data, "results": web.get("results", [])}
        return data


class NewsResult(BaseModel):
    title: str = ""
    url: Optional[str] = None
    description: str = ""
    age: Optional[str] = None
    thumbnail_url: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _extract_thumbnail(cls, data: Any) -> Any:
        if isinstance(data, dict) and "thumbnail_url" not in data:
            t = data.get("thumbnail")
            if isinstance(t, dict):
                data = {**data, "thumbnail_url": t.get("original") or t.get("src")}
        return data


class NewsSearchOperators(BaseModel):
    applied: bool = False
    cleaned_query: str = ""
    sites: list[str] = []


class NewsSearchQuery(BaseModel):
    original: str = ""
    altered: Optional[str] = None
    cleaned: Optional[str] = None
    spellcheck_off: bool = False
    show_strict_warning: bool = False
    search_operators: Optional[NewsSearchOperators] = None


class NewsSearchResponse(BaseModel):
    type: str = "news"
    query: NewsSearchQuery = NewsSearchQuery()
    results: list[NewsResult] = []

    @model_validator(mode="before")
    @classmethod
    def _flatten_news(cls, data: Any) -> Any:
        if isinstance(data, dict) and "results" not in data:
            news = data.get("news")
            if isinstance(news, dict):
                data = {**data, "results": news.get("results", [])}
        return data


class SuggestResponse(BaseModel):
    query: str
    suggestions: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _extract_suggestions(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "suggestions" in data:
            return data
        suggestions: list[str] = []
        for item in data.get("results", []):
            if isinstance(item, str):
                suggestions.append(item)
            elif isinstance(item, dict):
                s = item.get("query") or item.get("suggestion") or ""
                if s:
                    suggestions.append(str(s))
        return {**data, "suggestions": suggestions}


class ShoppingSearchResult(BaseModel):
    query: str
    sources: list[str] = []
    results: list[WebSearchResult] = []
