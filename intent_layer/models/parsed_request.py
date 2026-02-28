from typing import Optional, List
from intent_layer.models.intent import Intent
from personalization.tone.models import Tone
from pydantic import BaseModel
from websearch.models.common_properties import CommonProperties
from websearch.models.search_type import SearchType

class QueryDetails(BaseModel):
    query_text: str = ""
    keywords: Optional[List[str]] = None
    language: Optional[str] = None
    search_type: Optional[SearchType] = None

class ParsedRequest(BaseModel):
    intent: Intent
    tone: Optional[Tone] = None
    query_details: Optional[QueryDetails] = None
    common_properties: Optional[CommonProperties] = None
    safety_flags: Optional[List[str]] = None
