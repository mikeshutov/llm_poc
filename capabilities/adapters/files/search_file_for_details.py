from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel

from assets.files.urls import static_file_url
from assets.files.models import FileChunkResult
from assets.files.repository.file_chunk_repository import FileChunkRepository
from llm.clients.embeddings import embed_text
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from agent_runtime.context import get_current_user_id
from capabilities.registry.constants import TOOL_NAME_SEARCH_FILE_FOR_DETAILS
from capabilities.registry.constants import TOOL_RESULT_TYPE_FILE_DETAILS


class SearchFileForDetailsArgs(BaseModel):
    file_id: str
    query: str


def _tool_result(result: list[FileChunkResult]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for file_result in result:
        evidence_view = EvidenceView(
            item_id=str(file_result.file_id),
            tool_name=TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
            title=file_result.file_name.strip() or "File Detail",
            summary=file_result.content.strip() or "Matched file content.",
            urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)]
            if (file_url := static_file_url(file_result.file_path))
            else [],
            source=TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
            entity_type=TOOL_RESULT_TYPE_FILE_DETAILS,
            raw_payload=file_result,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
    args_schema=SearchFileForDetailsArgs,
    description="""
Search within one known uploaded file for content relevant to a query. Returns the most relevant chunks in document order.
Use file_search first when the relevant file_id is not known.


""",
)
def search_file_for_details(file_id: str, query: str) -> ToolResult:
    try:
        parsed_id = UUID(file_id)
    except ValueError:
        return ToolResult.error(f"Invalid file_id '{file_id}'. Use file_search to obtain a valid file_id first.")

    embedded_query = embed_text(query)
    results = FileChunkRepository().search_file_via_chunks(
        query_embedding=embedded_query,
        file_id=parsed_id,
        user_id=get_current_user_id(),
    )
    return _tool_result(
        results
    )
