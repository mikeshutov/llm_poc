from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WikipediaSearchResult(BaseModel):
    title: str
    description: str = ""
    url: str


class WikipediaPageSummary(BaseModel):
    title: str
    summary: str = ""
    url: str
    page_id: Optional[int] = None
