import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI()

def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        text = " "  # minimal non-empty input but probably want to just return to avoid making the call in general
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding