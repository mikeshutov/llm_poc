from openai import OpenAI

from common.model_constants import EMBEDDING_MODEL

client = OpenAI()


def embed_text(text: str) -> list[float]:
    text = (text or "").strip() or " "
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
