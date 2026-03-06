from pydantic import BaseModel

from integrations.models.search_type import SearchType


class GeneralSearchQuery(BaseModel):
    search_query: str
    search_type: SearchType
