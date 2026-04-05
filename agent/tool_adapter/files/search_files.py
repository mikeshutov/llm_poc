from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from files.repository.file_chunk_repository import FileChunkRepository, FileTypeFilter
from llm.clients.embeddings import embed_text

class SearchFilesArgs(BaseModel):
    query: str
    file_type: Optional[FileTypeFilter] = None

@tool(
    "search_files",
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
def search_files(query: str, file_type: Optional[FileTypeFilter] = None) -> list[dict]:
    embedded_query = embed_text(query)
    results = FileChunkRepository().search_file_via_chunks(
        query_embedding=embedded_query,
        file_type=file_type,
    )
    return [{"file_id": str(r.file_id), "file_name": r.file_name, "file_path": r.file_path, "top_chunk": r.content} for r in results]
