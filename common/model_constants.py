import os

AVAILABLE_CHAT_MODELS = (
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o3-pro",
    "o3-mini",
    "o4-mini",
    "o1",
    "o1-pro",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_ENCODING = "cl100k_base"
