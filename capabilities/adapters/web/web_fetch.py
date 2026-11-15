from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from integrations.web_fetch import WebFetchClient, WebFetchError
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from capabilities.registry.constants import TOOL_NAME_WEB_FETCH, TOOL_RESULT_TYPE_WEB_PAGE


class WebFetchArgs(BaseModel):
    url: str = Field(..., min_length=1, description="The public HTTP or HTTPS URL to fetch.")


_client = WebFetchClient()


@tool(
    TOOL_NAME_WEB_FETCH,
    args_schema=WebFetchArgs,
    description="Fetch a public web page and return its title, description, and readable text. Use this when the user provides a URL or asks for details from a specific page.",
)
def web_fetch(url: str) -> ToolResult:
    try:
        page = _client.fetch(url)
    except (WebFetchError, ValueError) as exc:
        return ToolResult.error(str(exc))

    evidence = EvidenceView(
        summary=page.description or page.text[:500],
        source=TOOL_NAME_WEB_FETCH,
        type=TOOL_RESULT_TYPE_WEB_PAGE,
        data={
            "item_id": page.url,
            "title": page.title or page.url,
            "urls": [EvidenceUrl(url=page.url, url_type=EvidenceUrlType.WEBSITE)],
        })
    return ToolResult(result=page, evidence=[evidence])




