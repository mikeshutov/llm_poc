from __future__ import annotations

from pydantic import BaseModel


class AstronomyPicture(BaseModel):
    title: str
    explanation: str
    date: str
    url: str
    media_type: str
