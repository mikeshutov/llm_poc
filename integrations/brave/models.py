from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WebSearchResult(BaseModel):
    title: str = ""
    url: Optional[str] = None
    description: str = ""
    image_url: Optional[str] = None


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult] = []


class NewsResult(BaseModel):
    title: str = ""
    url: Optional[str] = None
    description: str = ""
    age: Optional[str] = None


class NewsSearchResponse(BaseModel):
    query: str
    results: list[NewsResult] = []


class SuggestResponse(BaseModel):
    query: str
    suggestions: list[str] = []


class ShoppingSearchResult(BaseModel):
    query: str
    sources: list[str] = []
    results: list[WebSearchResult] = []
