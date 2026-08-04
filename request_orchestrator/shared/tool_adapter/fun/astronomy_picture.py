from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.nasa import NasaClient, AstronomyPicture

_client = NasaClient()


class GetAstronomyPictureArgs(BaseModel):
    date: Optional[str] = Field(
        default=None,
        description="Date in YYYY-MM-DD format to retrieve the astronomy picture for. Defaults to today.",
    )


@tool(
    "get_astronomy_picture",
    args_schema=GetAstronomyPictureArgs,
    description="""
Get NASA's Astronomy Picture of the Day (APOD) with title, explanation, and image URL.

Optional fields:
- date (string): date in YYYY-MM-DD format. Defaults to today's picture.

Example valid calls:
{}
{"date": "2024-01-15"}
{"date": "2023-07-04"}
""",
)
def get_astronomy_picture(date: str | None = None) -> AstronomyPicture | str:
    try:
        return _client.get_apod(date=date)
    except Exception as e:
        return f"NASA APOD API error: {e}"
