from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from files.urls import static_file_url
from files.models import FileChunkResult
from files.repository.file_chunk_repository import FileChunkRepository, FileTypeFilter
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_SEARCH_FILES
from tool.constants import TOOL_RESULT_TYPE_FILE_RESULTS


class SearchFilesArgs(BaseModel):
    query: str
    file_type: Optional[FileTypeFilter] = None


class SearchFilesMetadata(BaseModel):
    top_chunk: str


def _tool_result(result: list[FileChunkResult]) -> ToolResult:
    evidence: list[EvidenceView] = []

    for file_result in result:
        metadata = SearchFilesMetadata(top_chunk=file_result.content)
        evidence_view = EvidenceView(
            item_id=str(file_result.file_id),
            tool_name=TOOL_NAME_SEARCH_FILES,
            title=file_result.file_name.strip() or "File Search Result",
            summary=file_result.content.strip() or "Matched uploaded file.",
            urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)]
            if (file_url := static_file_url(file_result.file_path))
            else [],
            source=TOOL_NAME_SEARCH_FILES,
            entity_type=TOOL_RESULT_TYPE_FILE_RESULTS,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=file_result,
        )
        evidence.append(evidence_view)

    return ToolResult(
        result=result,

        evidence=evidence,
    )


@tool(
    TOOL_NAME_SEARCH_FILES,
    args_schema=SearchFilesArgs,
    description="""
Discover uploaded files by semantic similarity to a query. Use file_search_details when a file_id is already known.



""",
)
def search_files(query: str, file_type: Optional[FileTypeFilter] = None) -> ToolResult:
    embedded_query = embed_text(query)
    results = FileChunkRepository().search_file_via_chunks(
        query_embedding=embedded_query,
        file_type=file_type,
        user_id=get_current_user_id(),
    )
    return _tool_result(
        results
    )
