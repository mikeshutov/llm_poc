from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel

from files.repository.file_chunk_repository import FileChunkRepository
from llm.clients.embeddings import embed_text


class SearchFileForDetailsArgs(BaseModel):
    file_id: str
    query: str


@tool(
    "search_file_for_details",
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
def search_file_for_details(file_id: str, query: str) -> list[dict]:
    try:
        parsed_id = UUID(file_id)
    except ValueError:
        return [{"error": f"Invalid file_id '{file_id}'. Use search_files to obtain a valid file_id first."}]
    
    embedded_query = embed_text(query)
    results = FileChunkRepository().search_file_via_chunks(
        query_embedding=embedded_query,
        file_id=parsed_id,
    )
    return [{"file_id": str(r.file_id), "file_name": r.file_name, "file_path": r.file_path, "content": r.content} for r in results]
