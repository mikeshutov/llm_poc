from common.model_constants import EMBEDDING_MODEL
from llm.clients.llm_client import get_openai_client


def embed_text(text: str) -> list[float]:
    text = (text or "").strip() or " "
    resp = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding
