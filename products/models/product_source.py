from enum import Enum


class ProductSource(str, Enum):
    DB = "db"
    RAG = "rag"
    WEB = "web"


class ProductSearchSource(str, Enum):
    INTERNAL = "internal"
    WEB = "web"
    BOTH = "both"
