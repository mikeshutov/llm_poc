from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from files.repository.file_repository import FileRepository


class GetFileByIdArgs(BaseModel):
    file_id: str = Field(..., description="The UUID of the file to retrieve.")


@tool(
    "get_file_by_id",
    args_schema=GetFileByIdArgs,
    description="""
Retrieve file metadata and a preview of its contents by file_id. Returns name, type, upload date, and the first chunk of content.
Use when a file_id is already known from context or a previous tool call.

Required fields:
- file_id (string): The UUID of the file to retrieve.

Example valid calls:
{"file_id": "3f2a1b4c-..."}
""",
)
def get_file_by_id(file_id: str) -> dict:
    try:
        parsed_id = UUID(file_id)
    except ValueError:
        return {"error": f"Invalid file_id '{file_id}'."}
    row = FileRepository().get_file_by_id(parsed_id)
    if not row:
        return {"error": f"No file found with id '{file_id}'."}
    return {
        "file_id": str(row["id"]),
        "file_name": row["file_name"],
        "file_type": row["file_type"],
        "uploaded_at": str(row["uploaded_at"]),
        "first_chunk": row.get("first_chunk"),
    }
