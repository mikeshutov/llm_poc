from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HnHit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    object_id: str = Field(alias="objectID")
    title: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    points: Optional[int] = None
    num_comments: Optional[int] = None
    created_at: Optional[str] = None
    story_text: Optional[str] = None
    tags: Optional[list[str]] = Field(default=None, alias="_tags")


class HnSearchResult(BaseModel):
    hits: list[HnHit] = []
    nb_hits: int = Field(default=0, alias="nbHits")
    page: int = 0
    nb_pages: int = Field(default=0, alias="nbPages")
