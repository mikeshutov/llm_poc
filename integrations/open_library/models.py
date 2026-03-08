from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BookDoc(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    title: str
    author_name: Optional[list[str]] = None
    first_publish_year: Optional[int] = None
    edition_count: Optional[int] = None
    isbn: Optional[list[str]] = None
    subject: Optional[list[str]] = None
    publisher: Optional[list[str]] = None
    language: Optional[list[str]] = None
    cover_i: Optional[int] = None


class BookSearchResult(BaseModel):
    num_found: int = Field(alias="numFound")
    start: int = 0
    docs: list[BookDoc] = []
