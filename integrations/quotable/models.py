from __future__ import annotations

from pydantic import BaseModel


class Quote(BaseModel):
    content: str
    author: str
    tags: list[str] = []
