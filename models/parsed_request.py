from typing import Optional
from models.intent import Intent
from models.product_query import ProductQuery
from pydantic import BaseModel

class ParsedRequest(BaseModel):
    intent: Intent
    product_query: Optional[ProductQuery] = None
