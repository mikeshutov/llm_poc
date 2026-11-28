from dataclasses import asdict
from unittest.mock import MagicMock
from uuid import uuid4


def test_context_query_filters_dedicated_column_but_audit_query_does_not(
    conversation_message_factory,
    conversation_repository_factory,
):
    repository, cursor = conversation_repository_factory()
    message = conversation_message_factory(
        kind="tool_result",
        visibility="internal",
        role="tool",
        compacted=True,
    )
    cursor.fetchall.return_value = [asdict(message)]

    assert repository.list_messages(message.roundtrip_id) == [message]
    audit_query = cursor.execute.call_args.args[0]
    assert "AND NOT compacted" not in audit_query
    assert "compacted" in audit_query.split("FROM")[0]

    repository.list_messages(message.roundtrip_id, exclude_compacted=True)
    query, parameters = cursor.execute.call_args.args
    assert "AND NOT compacted" in query
    assert "metadata->>" not in query
    assert parameters == [message.roundtrip_id]


def test_marks_exact_result_ids_and_their_parent_calls_without_deleting(conversation_repository_factory, message_tool_repository_factory):
    repository, cursor = conversation_repository_factory()
    tools = message_tool_repository_factory(repository=repository)
    result_ids = [uuid4(), uuid4()]

    tools.mark_compacted(result_ids)

    query, parameters = cursor.execute.call_args.args
    assert parameters == (result_ids,)
    assert "WHERE id = ANY(%s) AND kind = 'tool_result'" in query
    assert "SET compacted = TRUE" in query
    assert "message.id = results.id OR message.id = results.parent_message_id" in query
    assert "message.roundtrip_id = results.roundtrip_id" in query
    assert "AND NOT message.compacted" in query
    assert "payload->>'compactable' IS DISTINCT FROM 'false'" in query
    assert "DELETE" not in query
    assert "metadata" not in query


def test_empty_compaction_does_not_query(conversation_repository_factory, message_tool_repository_factory):
    repository, cursor = conversation_repository_factory()
    tools = message_tool_repository_factory(repository=repository)

    tools.mark_compacted([])

    repository._conn.cursor.assert_not_called()


def test_append_persists_explicit_compacted_flag(
    conversation_message_factory,
    conversation_repository_factory,
):
    repository, cursor = conversation_repository_factory()
    message = conversation_message_factory(
        kind="tool_result",
        visibility="internal",
        role="tool",
        compacted=True,
    )
    cursor.fetchone.return_value = asdict(message)

    saved = repository.append_message(
        message.roundtrip_id, kind=message.kind, role=message.role, visibility="internal", compacted=True,
    )

    query, parameters = cursor.execute.call_args.args
    assert "subagent_name, compacted, embedding" in query
    assert parameters[10] is True
    assert query.count("%s") == len(parameters)
    assert saved.compacted is True
    assert saved.metadata == {}


def test_tool_lookup_queries_ids_directly_and_preserves_requested_order(
    conversation_message_factory,
    conversation_repository_factory,
    message_tool_repository_factory,
):
    repository, cursor = conversation_repository_factory()
    tools = message_tool_repository_factory(repository=repository)
    first = conversation_message_factory(
        kind="tool_result",
        visibility="internal",
        role="tool",
        compacted=True,
    )
    second = conversation_message_factory(
        roundtrip_id=first.roundtrip_id,
        sequence=1,
        kind="tool_result",
        visibility="internal",
        role="tool",
    )
    cursor.fetchall.return_value = [asdict(second), asdict(first)]
    ids = [first.id, uuid4(), second.id]

    assert tools._tool_messages_for_ids(ids) == [first, second]

    query, parameters = cursor.execute.call_args.args
    assert "id = ANY(%s)" in query
    assert "AND NOT compacted" not in query  # Evidence/audit reads include compacted records.
    assert parameters == (ids,)
    assert cursor.execute.call_count == 1


def test_request_count_is_a_single_sql_count(conversation_repository_factory, message_tool_repository_factory):
    repository, cursor = conversation_repository_factory()
    tools = message_tool_repository_factory(repository=repository)
    cursor.fetchone.return_value = {"count": 2}
    roundtrip_id = uuid4()

    assert tools.count_request_hash(roundtrip_id, "signature") == 2

    query, parameters = cursor.execute.call_args.args
    assert "COUNT(*)" in query
    assert "payload->>'request_hash' = %s" in query
    assert parameters == (roundtrip_id, "signature")


def test_tool_messages_record_the_executing_agent_not_the_tool_name(message_tool_repository_factory, plan_step_factory):
    tools = message_tool_repository_factory(repository=MagicMock())
    tools.append_tool_call(
        uuid4(), None, plan_step_factory(id="search", tool="web_search", plan="Search", args={}),
        is_subagent=True, subagent_name="research",
    )

    writes = tools._repo.append_message.call_args_list
    assert len(writes) == 2
    assert all(write.kwargs["subagent_name"] == "research" for write in writes)
