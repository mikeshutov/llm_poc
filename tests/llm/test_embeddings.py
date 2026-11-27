from hashlib import sha256
from unittest.mock import Mock

import pytest

from common.config import UTF8_ENCODING
from llm.clients import embeddings
from llm.repository import embedding_cache_repository as repository
from llm.repository.repo_factory import get_embedding_cache_repo


@pytest.fixture
def embedding_dependencies(monkeypatch, embedding_client_factory):
    client = embedding_client_factory()
    provider = Mock(return_value=client)
    cache = Mock(spec=repository.EmbeddingCacheRepository)
    lookup = cache.get
    lookup.return_value = None
    store = cache.put
    log = Mock()
    monkeypatch.setattr(embeddings, "get_openai_client", provider)
    monkeypatch.setattr(embeddings, "get_embedding_cache_repo", lambda: cache)
    monkeypatch.setattr(embeddings, "_record_embedding_log", log)
    return client, provider, lookup, store, log


@pytest.mark.parametrize("text,normalized", [("  café  ", "café"), ("", " "), (" \n ", " ")])
def test_embedding_miss_stores_normalized_input_hash(embedding_dependencies, text, normalized):
    client, _, lookup, store, log = embedding_dependencies
    result = embeddings.embed_text(text)
    key = dict(model=embeddings.EMBEDDING_MODEL, input_hash=sha256(normalized.encode(UTF8_ENCODING)).hexdigest())
    lookup.assert_called_once_with(**key)
    client.embeddings.create.assert_called_once_with(model=embeddings.EMBEDDING_MODEL, input=normalized)
    store.assert_called_once_with(**key, embedding=result)
    assert log.call_args.kwargs["cache_error"] == ""


def test_embedding_hit_skips_provider(embedding_dependencies):
    _, provider, lookup, store, log = embedding_dependencies
    lookup.return_value = [0.25, 0.75]
    assert embeddings.embed_text("cached") == [0.25, 0.75]
    provider.assert_not_called()
    store.assert_not_called()
    assert log.call_args.kwargs["cache_hit"] is True


def test_embedding_cache_reuses_across_callers_and_separates_models_and_inputs(
    monkeypatch, embedding_dependencies,
):
    client, _, lookup, store, _ = embedding_dependencies
    entries = {}
    lookup.side_effect = lambda **key: entries.get((key["model"], key["input_hash"]))
    store.side_effect = lambda **value: entries.setdefault(
        (value["model"], value["input_hash"]), value["embedding"],
    )
    embeddings.embed_text("same", purpose="agent_loading")
    embeddings.embed_text(" same ", purpose="capability_discovery")
    assert client.embeddings.create.call_count == 1
    embeddings.embed_text("different")
    monkeypatch.setattr(embeddings, "EMBEDDING_MODEL", "another-model")
    embeddings.embed_text("same")
    assert client.embeddings.create.call_count == 3
    assert len(entries) == 3


@pytest.mark.parametrize("failing_operation", ["lookup", "store"])
def test_embedding_cache_failure_does_not_block_provider(embedding_dependencies, failing_operation):
    client, _, lookup, store, log = embedding_dependencies
    (lookup if failing_operation == "lookup" else store).side_effect = RuntimeError("database unavailable")
    assert embeddings.embed_text("query") == [1.0, 0.0]
    client.embeddings.create.assert_called_once()
    assert log.call_args.kwargs["cache_error"] == "database unavailable"
    if failing_operation == "lookup":
        store.assert_not_called()


def test_failed_embedding_is_not_cached(embedding_dependencies):
    client, _, _, store, log = embedding_dependencies
    client.embeddings.create.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        embeddings.embed_text("query")
    store.assert_not_called()
    assert log.call_args.kwargs["error"] == "provider unavailable"


@pytest.mark.parametrize("row,expected", [(None, None), ({"embedding": [0.25, 0.75]}, [0.25, 0.75])])
def test_repository_lookup(monkeypatch, database_connection_factory, row, expected):
    connection, cursor = database_connection_factory(row=row)
    monkeypatch.setattr(repository, "get_connection", lambda: connection)
    assert repository.EmbeddingCacheRepository().get(model="model", input_hash="hash") == expected
    sql, params = cursor.execute.call_args.args
    assert "WHERE model = %s AND input_hash = %s" in sql
    assert "created_at" not in sql
    assert params == ("model", "hash")
    connection.close.assert_called_once()


def test_repository_insert_is_conflict_safe(monkeypatch, database_connection_factory):
    connection, cursor = database_connection_factory()
    monkeypatch.setattr(repository, "get_connection", lambda: connection)
    repository.EmbeddingCacheRepository().put(model="model", input_hash="hash", embedding=[0.25, 0.75])
    sql, params = cursor.execute.call_args.args
    assert "ON CONFLICT (model, input_hash) DO NOTHING" in sql
    assert params == ("model", "hash", [0.25, 0.75])
    connection.close.assert_called_once()


@pytest.mark.parametrize("operation", ["get", "put"])
def test_repository_closes_connection_on_error(monkeypatch, database_connection_factory, operation):
    connection, cursor = database_connection_factory()
    cursor.execute.side_effect = RuntimeError("query failed")
    monkeypatch.setattr(repository, "get_connection", lambda: connection)
    with pytest.raises(RuntimeError, match="query failed"):
        if operation == "get":
            repository.EmbeddingCacheRepository().get(model="model", input_hash="hash")
        else:
            repository.EmbeddingCacheRepository().put(model="model", input_hash="hash", embedding=[1.0])
    connection.close.assert_called_once()


def test_repository_factory_does_not_open_connection(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(repository, "get_connection", connect)
    assert isinstance(get_embedding_cache_repo(), repository.EmbeddingCacheRepository)
    connect.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_repository_uses_but_does_not_close_injected_connection(
    monkeypatch, database_connection_factory, fail,
):
    connection, cursor = database_connection_factory(row={"embedding": [1.0]})
    connect = Mock()
    monkeypatch.setattr(repository, "get_connection", connect)
    cache = repository.EmbeddingCacheRepository(conn=connection)
    if fail:
        cursor.execute.side_effect = RuntimeError("query failed")
        with pytest.raises(RuntimeError, match="query failed"):
            cache.get(model="model", input_hash="hash")
    else:
        assert cache.get(model="model", input_hash="hash") == [1.0]
        cache.put(model="model", input_hash="hash", embedding=[1.0])
    connect.assert_not_called()
    connection.close.assert_not_called()
