from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel

from assets.files.urls import static_file_url
from assets.models import AssetChunkResult, AssetType
from assets.repository.asset_chunk_repository import AssetChunkRepository
from llm.clients.embeddings import embed_text
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from agent_runtime.context import get_current_user_id
from capabilities.registry.constants import TOOL_NAME_SEARCH_ASSET_DETAILS, TOOL_RESULT_TYPE_ASSET_DETAILS


class SearchAssetDetailsArgs(BaseModel):
    asset_id: str
    query: str
    asset_type: AssetType = AssetType.FILE


def _tool_result(result: list[AssetChunkResult]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for file_result in result:
        evidence_view = EvidenceView(
            item_id=str(file_result.asset_id),
            tool_name=TOOL_NAME_SEARCH_ASSET_DETAILS,
            title=file_result.asset_name.strip() or "Asset Detail",
            summary=file_result.content.strip() or "Matched asset content.",
            urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)] if (file_url := static_file_url(file_result.storage_path or "")) else ([EvidenceUrl(url=file_result.url, url_type=EvidenceUrlType.WEBSITE)] if file_result.url else []),
            source=TOOL_NAME_SEARCH_ASSET_DETAILS,
            entity_type=TOOL_RESULT_TYPE_ASSET_DETAILS,
            raw_payload=file_result,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_SEARCH_ASSET_DETAILS,
    args_schema=SearchAssetDetailsArgs,
    description="""
Search within one known uploaded file for content relevant to a query. Returns the most relevant chunks in document order.
Use asset_search first when the relevant asset_id is not known.


""",
)
def search_asset_details(asset_id: str, query: str, asset_type: AssetType = AssetType.FILE) -> ToolResult:
    try:
        parsed_id = UUID(asset_id)
    except ValueError:
        return ToolResult.error(f"Invalid asset_id '{asset_id}'. Use asset_search to obtain a valid asset_id first.")

    embedded_query = embed_text(query)
    results = AssetChunkRepository().search_assets(
        query_embedding=embedded_query,
        asset_id=parsed_id,
        asset_type=asset_type.value,
        user_id=get_current_user_id(),
    )
    return _tool_result([AssetChunkResult.model_validate(result) for result in results])
