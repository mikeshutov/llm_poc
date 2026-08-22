from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel

from files.repository.file_chunk_repository import FileChunkRepository
from llm.clients.embeddings import embed_text
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from tool.constants import TOOL_NAME_SEARCH_FILE_FOR_DETAILS
from tool.constants import TOOL_RESULT_TYPE_FILE_DETAILS


class SearchFileForDetailsArgs(BaseModel):
    file_id: str
    query: str


class SearchFileForDetailsMetadata(BaseModel):
    file_name: str | None = None


def _tool_result(result: list[dict]) -> ToolResult:
    hydrated_evidence: list[HydratedEvidence] = []
    evidence_views: list[EvidenceView] = []
    for file_result in result:
        metadata = SearchFileForDetailsMetadata(
            file_name=str(file_result.get("file_name", "")).strip() or None,
        )
        hydrated = HydratedEvidence(
            item_id=str(file_result.get("file_id", "")),
            tool_name=TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
            title=str(file_result.get("file_name", "")).strip() or "File Detail",
            summary=str(file_result.get("content", "")).strip() or "Matched file content.",
            source=TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
            entity_type=TOOL_RESULT_TYPE_FILE_DETAILS,
            metadata=metadata.model_dump(exclude_none=True),
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
    return ToolResult(result=result, evidence_views=evidence_views, hydrated_evidence=hydrated_evidence)




@tool(
    TOOL_NAME_SEARCH_FILE_FOR_DETAILS,
    args_schema=SearchFileForDetailsArgs,
    description="""
Search within a specific uploaded file for content relevant to a query. Returns the most relevant chunks in document order.
Use search_files first to find relevant files and get their file_id, then use this tool to retrieve specific content.

Required fields:
- file_id (string): The UUID of the file to search within, obtained from search_files.
- query (string): Natural language description of what you are looking for within the file.

Example valid calls:
{"file_id": "3f2a1b4c-...", "query": "work experience at Acme Corp"}
""",
)
def search_file_for_details(file_id: str, query: str) -> ToolResult:
    try:
        parsed_id = UUID(file_id)
    except ValueError:
        return ToolResult.error(f"Invalid file_id '{file_id}'. Use search_files to obtain a valid file_id first.")

    embedded_query = embed_text(query)
    results = FileChunkRepository().search_file_via_chunks(
        query_embedding=embedded_query,
        file_id=parsed_id,
        user_id=get_current_user_id(),
    )
    return _tool_result(
        [{"file_id": str(r.file_id), "file_name": r.file_name, "file_path": r.file_path, "content": r.content} for r in results]
    )
