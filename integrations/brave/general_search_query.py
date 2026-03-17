from pydantic import BaseModel

from integrations.brave.search_type import SearchType


class GeneralSearchQuery(BaseModel):
    search_query: str
    search_type: SearchType
