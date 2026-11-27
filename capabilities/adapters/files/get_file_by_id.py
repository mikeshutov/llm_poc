from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from assets.repository.asset_chunk_repository import AssetChunkRepository
from assets.repository.asset_repository import AssetRepository
from assets.files.urls import static_file_url
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from agent_runtime.context import current_agent_state
from capabilities.adapters.files.result_models import GetFileByIdResult
from capabilities.registry.constants import TOOL_NAME_GET_FILE_BY_ID
from capabilities.registry.constants import TOOL_RESULT_TYPE_FILE


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
        summary=(result.first_chunk or "").strip() or "Retrieved file metadata and preview.",
        source=TOOL_NAME_GET_FILE_BY_ID,
        type=TOOL_RESULT_TYPE_FILE,
        metadata=metadata.model_dump(exclude_none=True),
        data={
            "item_id": (result.file_id or "").strip(),
            "title": (result.file_name or "").strip() or "File",
            "urls": [EvidenceUrl(url=file_url, url_type=EvidenceUrlType.WEBSITE)]
        if (file_url := static_file_url(result.file_path or ""))
        else [],
        })
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
    active_state = current_agent_state.get()
    active_user_id = active_state.execution_context.user_profile.user_id if active_state is not None else None
    file = AssetRepository().get_file(parsed_id, user_id=active_user_id)
    if not file:
        return ToolResult.error(f"No file found with id '{file_id}'.")
    return _tool_result(
        GetFileByIdResult(
            file_id=str(file.id),
            file_name=file.name,
            file_path=file.storage_path,
            file_type=file.mime_type,
            uploaded_at=file.created_at.isoformat(),
            first_chunk=AssetChunkRepository().get_first(file.id, user_id=active_user_id),
        )
    )




