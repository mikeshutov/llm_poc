from dataclasses import asdict
from uuid import uuid4

from capabilities.registry.repository.message_repository import MessageToolRepository


def test_tool_message_lookup_queries_requested_results_in_input_order(
    database_connection_factory, conversation_message_factory,
):
    first = conversation_message_factory(kind="tool_result", role="tool")
    second = conversation_message_factory(kind="tool_result", role="tool")
    connection, cursor = database_connection_factory(rows=[asdict(second), asdict(first)])
    repository = object.__new__(MessageToolRepository)
    repository._repo = type("Repository", (), {"_conn": connection})()
    ids = [first.id, uuid4(), second.id, first.id]

    messages = repository._tool_messages_for_ids(ids)

    assert [message.id for message in messages] == [first.id, second.id, first.id]
    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "id = ANY(%s)" in sql
    assert "kind = 'tool_result'" in sql
    assert params == (ids,)


def test_empty_tool_message_lookup_does_not_query():
    repository = object.__new__(MessageToolRepository)
    assert repository._tool_messages_for_ids([]) == []


def test_request_hash_count_is_filtered_in_database(database_connection_factory):
    connection, cursor = database_connection_factory(row={"count": 3})
    repository = object.__new__(MessageToolRepository)
    repository._repo = type("Repository", (), {"_conn": connection})()
    roundtrip_id = uuid4()

    assert repository.count_request_hash(roundtrip_id, "request-hash") == 3

    sql, params = cursor.execute.call_args.args
    assert "COUNT(*)" in sql
    assert "kind = 'tool_result'" in sql
    assert params == (roundtrip_id, "request-hash")


def test_tool_messages_record_the_executing_agent_name(
    recording_conversation_repository_factory, plan_step_factory,
):
    conversation_repository = recording_conversation_repository_factory()
    repository = object.__new__(MessageToolRepository)
    repository._repo = conversation_repository

    repository.append_tool_call(
        uuid4(), None, plan_step_factory(tool="web_search"),
        is_subagent=True, subagent_name="research",
    )

    assert len(conversation_repository.messages) == 2
    assert all(message["is_subagent"] for message in conversation_repository.messages)
    assert all(message["subagent_name"] == "research" for message in conversation_repository.messages)
