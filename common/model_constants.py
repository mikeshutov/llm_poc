import os

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_ENCODING = "cl100k_base"
