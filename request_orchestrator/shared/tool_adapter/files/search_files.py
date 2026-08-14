from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from files.repository.file_chunk_repository import FileChunkRepository, FileTypeFilter
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_SEARCH_FILES
from tool.constants import TOOL_RESULT_TYPE_FILE_RESULTS


class SearchFilesArgs(BaseModel):
    query: str
    file_type: Optional[FileTypeFilter] = None


def _tool_result(result: list[dict]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []

    for file_result in result:
        hydrated = HydratedEvidence(
            item_id=str(file_result.get("file_id", "")),
            tool_name=TOOL_NAME_SEARCH_FILES,
            title=str(file_result.get("file_name", "")).strip() or "File Search Result",
            summary=str(file_result.get("top_chunk", "")).strip() or "Matched uploaded file.",
            source=TOOL_NAME_SEARCH_FILES,
            entity_type=TOOL_RESULT_TYPE_FILE_RESULTS,
            metadata={
                "file_name": file_result.get("file_name", ""),
                "file_path": file_result.get("file_path", ""),
                "top_chunk": file_result.get("top_chunk", ""),
            },
            raw_payload=file_result,
        )
        hydrated_evidence.append(hydrated)
        evidence_views.append(
            EvidenceView(
                item_id=hydrated.item_id,
                title=hydrated.title,
                summary=hydrated.summary,
                metadata=dict(hydrated.metadata),
            )
        )

    return ToolResult(
        result=result,
        evidence_views=evidence_views,
        hydrated_evidence=hydrated_evidence,
    )


@tool(
    TOOL_NAME_SEARCH_FILES,
    args_schema=SearchFilesArgs,
    description="""
Search uploaded files by semantic similarity to a query. Returns file names that are relevant to the query.

Required fields:
- query (string): Natural language description of what you are looking for.

Optional fields:
- file_type (string): Filter by file type. Use "image" for images (jpg, jpeg, png, webp) or "text" for documents (pdf, txt, docx).

Example valid calls:
{"query": "work experience at tech companies"}
{"query": "profile photo", "file_type": "image"}
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
        [{"file_id": str(r.file_id), "file_name": r.file_name, "file_path": r.file_path, "top_chunk": r.content} for r in results]
    )
