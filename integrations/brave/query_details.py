from typing import Optional, List
from pydantic import BaseModel
from integrations.brave.search_type import SearchType


class QueryDetails(BaseModel):
    query_text: str = ""
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    search_type: Optional[SearchType] = None
