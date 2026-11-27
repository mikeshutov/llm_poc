from __future__ import annotations

from hashlib import sha256
from time import perf_counter

from common.config import EMBEDDING_MODEL, UTF8_ENCODING
from llm.clients.llm_client import get_openai_client
from llm.repository.repo_factory import get_embedding_cache_repo


def embed_text(
    text: str,
    *,
    purpose: str = "embedding",
    node_name: str = "embedding",
) -> list[float]:
    text = (text or "").strip() or " "
    input_hash = sha256(text.encode(UTF8_ENCODING)).hexdigest()
    started_at = perf_counter()
    cache_error = ""
    repository = get_embedding_cache_repo()
    try:
        embedding = repository.get(model=EMBEDDING_MODEL, input_hash=input_hash)
    except Exception as exc:
        embedding = None
        cache_error = str(exc)
    if embedding is not None:
        _record_embedding_log(
            node_name=node_name,
            purpose=purpose,
            text=text,
            duration_ms=int((perf_counter() - started_at) * 1000),
            dimensions=len(embedding),
            cache_hit=True,
        )
        return embedding

    try:
        resp = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
        embedding = resp.data[0].embedding
    except Exception as exc:
        _record_embedding_log(
            node_name=node_name,
            purpose=purpose,
            text=text,
            duration_ms=int((perf_counter() - started_at) * 1000),
            error=str(exc),
            cache_error=cache_error,
        )
        raise

    if not cache_error:
        try:
            repository.put(model=EMBEDDING_MODEL, input_hash=input_hash, embedding=embedding)
        except Exception as exc:
            cache_error = str(exc)

    _record_embedding_log(
        node_name=node_name,
        purpose=purpose,
        text=text,
        duration_ms=int((perf_counter() - started_at) * 1000),
        dimensions=len(embedding),
        cache_error=cache_error,
    )
    return embedding


def _record_embedding_log(
    *,
    node_name: str,
    purpose: str,
    text: str,
    duration_ms: int,
    dimensions: int | None = None,
    error: str = "",
    cache_hit: bool = False,
    cache_error: str = "",
) -> None:
    # Embeddings are also used by repository and tool code. Resolve logging
    # lazily so standalone scripts and startup-time model setup do not create
    # a logging dependency or fail when no roundtrip is active.
    from common.logging import record_node_log
    from agent_runtime.context import get_current_roundtrip_id

    record_node_log(
        node_name=node_name,
        stage="embedding",
        event_type="embedding_call",
        level="error" if error else ("warning" if cache_error else "info"),
        message=(
            "Embedding call failed" if error else
            "Embedding cache hit" if cache_hit else "Embedding call completed"
        ),
        roundtrip_id=get_current_roundtrip_id(),
        duration_ms=duration_ms,
        payload={
            "model": EMBEDDING_MODEL,
            "purpose": purpose,
            "input_preview": text[:500],
            "input_length": len(text),
            "input_hash": sha256(text.encode(UTF8_ENCODING)).hexdigest(),
            "dimensions": dimensions,
            "cache_hit": cache_hit,
            "cache_error": cache_error[:2_000],
            "error": error[:2_000],
        },
    )
