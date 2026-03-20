import os

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
