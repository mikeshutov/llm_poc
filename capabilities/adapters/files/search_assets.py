from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from assets.files.urls import static_file_url
from assets.models import AssetChunkResult, AssetContentType, AssetType
from assets.repository.asset_chunk_repository import AssetChunkRepository
from llm.clients.embeddings import embed_text
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from agent_runtime.context import get_current_user_id
from capabilities.registry.constants import TOOL_NAME_SEARCH_ASSETS, TOOL_RESULT_TYPE_ASSET_RESULTS


class SearchAssetsArgs(BaseModel):
    query: str
    asset_type: Optional[AssetType] = None
    content_type: Optional[AssetContentType] = None


class SearchAssetsMetadata(BaseModel):
    top_chunk: str


def _tool_result(result: list[AssetChunkResult]) -> ToolResult:
    evidence: list[EvidenceView] = []

    for asset_result in result:
        metadata = SearchAssetsMetadata(top_chunk=asset_result.content)
        evidence_view = EvidenceView(
            item_id=str(asset_result.asset_id),
            tool_name=TOOL_NAME_SEARCH_ASSETS,
            title=asset_result.asset_name.strip() or "Asset Search Result",
            summary=asset_result.content.strip() or "Matched asset.",
            urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)] if (file_url := static_file_url(asset_result.storage_path or "")) else ([EvidenceUrl(url=asset_result.url, url_type=EvidenceUrlType.WEBSITE)] if asset_result.url else []),
            source=TOOL_NAME_SEARCH_ASSETS,
            entity_type=TOOL_RESULT_TYPE_ASSET_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=asset_result,
        )
        evidence.append(evidence_view)

    return ToolResult(
        result=result,

        evidence=evidence,
    )


@tool(
    TOOL_NAME_SEARCH_ASSETS,
    args_schema=SearchAssetsArgs,
    description="""
Search assets by semantic similarity to a query. Use asset_search_details when an asset_id is already known.
If asset_type is omitted, search all indexed asset types visible to the current user.
Use asset_type to restrict the search to file or http_page assets, and content_type to restrict file content to text or image.
""",
)
def search_assets(query: str, asset_type: Optional[AssetType] = None, content_type: Optional[AssetContentType] = None) -> ToolResult:
    embedded_query = embed_text(query)
    results = AssetChunkRepository().search_assets(
        query_embedding=embedded_query,
        asset_type=asset_type.value if asset_type else None,
        content_type=content_type,
        user_id=get_current_user_id(),
    )
    return _tool_result([AssetChunkResult.model_validate(result) for result in results])
