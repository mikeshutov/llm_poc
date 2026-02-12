from enum import Enum


class ProductSource(str, Enum):
    DB = "db"
    RAG = "rag"
    WEB = "web"
