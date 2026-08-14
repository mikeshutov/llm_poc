from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.nasa import NasaClient
from integrations.nasa.models import AstronomyPicture
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, HydratedEvidence, ToolResult
from tool.constants import TOOL_NAME_GET_ASTRONOMY_PICTURE
from tool.constants import TOOL_RESULT_TYPE_ASTRONOMY_PICTURE

_client = NasaClient()


class GetAstronomyPictureArgs(BaseModel):
    date: Optional[str] = Field(
        default=None,
        description="Date in YYYY-MM-DD format to retrieve the astronomy picture for. Defaults to today.",
    )


def _tool_result(result: AstronomyPicture) -> ToolResult:
    url = result.url.strip()
    hydrated = HydratedEvidence(
        item_id=f"{result.date}:{result.title}",
        tool_name=TOOL_NAME_GET_ASTRONOMY_PICTURE,
        title=result.title,
        summary=result.explanation,
        urls=[EvidenceUrl(url=url, url_type="website")] if url else [],
        image_url=url if result.media_type == "image" else "",
        published_at=result.date,
        source=TOOL_NAME_GET_ASTRONOMY_PICTURE,
        entity_type=TOOL_RESULT_TYPE_ASTRONOMY_PICTURE,
        metadata={"media_type": result.media_type},
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence_views=[
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        ],
        hydrated_evidence=[hydrated],
    )




@tool(
    TOOL_NAME_GET_ASTRONOMY_PICTURE,
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
def get_astronomy_picture(date: str | None = None) -> ToolResult:
    try:
        return _tool_result(_client.get_apod(date=date))
    except Exception as e:
        return ToolResult.error(f"NASA APOD API error: {e}")
