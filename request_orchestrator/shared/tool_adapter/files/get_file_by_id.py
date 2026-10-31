from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from files.repository.file_chunk_repository import FileChunkRepository
from files.repository.file_repository import FileRepository
from files.urls import static_file_url
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from request_orchestrator.shared.runtime_context import get_current_user_id
from request_orchestrator.shared.tool_adapter.files.result_models import GetFileByIdResult
from tool.constants import TOOL_NAME_GET_FILE_BY_ID
from tool.constants import TOOL_RESULT_TYPE_FILE


class GetFileByIdArgs(BaseModel):
    file_id: str = Field(..., description="The UUID of the file to retrieve.")


class GetFileByIdMetadata(BaseModel):
    file_type: str | None = None
    uploaded_at: str


def _tool_result(result: GetFileByIdResult) -> ToolResult:
    metadata = GetFileByIdMetadata(
        file_type=result.file_type,
        uploaded_at=result.uploaded_at,
    )
    evidence_view = EvidenceView(
        item_id=(result.file_id or "").strip(),
        tool_name=TOOL_NAME_GET_FILE_BY_ID,
        title=(result.file_name or "").strip() or "File",
        summary=(result.first_chunk or "").strip() or "Retrieved file metadata and preview.",
        urls=[EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)]
        if (file_url := static_file_url(result.file_path or ""))
        else [],
        source=TOOL_NAME_GET_FILE_BY_ID,
        entity_type=TOOL_RESULT_TYPE_FILE,
        llm_metadata=metadata.model_dump(exclude_none=True),
        raw_payload=result,
    )
    return ToolResult(
        result=result,
        evidence=[evidence_view],
    )




@tool(
    TOOL_NAME_GET_FILE_BY_ID,
    args_schema=GetFileByIdArgs,
    description="""
Retrieve file metadata and a short preview by file_id. Use file.read for the complete file content.
Use when a file_id is already known from context or a previous tool call.


""",
)
def get_file_by_id(file_id: str) -> ToolResult:
    try:
        parsed_id = UUID(file_id)
    except ValueError:
        return ToolResult.error(f"Invalid file_id '{file_id}'.")
    file = FileRepository().get_file_by_id(parsed_id, user_id=get_current_user_id())
    if not file:
        return ToolResult.error(f"No file found with id '{file_id}'.")
    return _tool_result(
        GetFileByIdResult(
            file_id=str(file.id),
            file_name=file.name,
            file_path=file.storage_path,
            file_type=file.mime_type,
            uploaded_at=file.created_at.isoformat(),
            first_chunk=FileChunkRepository().get_first_chunk(file.id, user_id=get_current_user_id()),
        )
    )
