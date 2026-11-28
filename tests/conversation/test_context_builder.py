from dataclasses import replace
from uuid import uuid4
from unittest.mock import patch

import pytest

from conversation.context_builder import build_roundtrip_context


def test_context_uses_ordered_visible_messages(context_repository_factory):
    repository = context_repository_factory()
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.content for message in context.messages] == ["Find boots.", "Here are boots."]
    assert context.conversation_summary == ""
    assert repository.message_queries == [(repository.roundtrip.id, True, True)]


def test_context_omits_marked_pairs_but_audit_reads_keep_them(
    conversation_tool_exchange_factory,
    context_repository_factory,
):
    repository = context_repository_factory()
    call, result = conversation_tool_exchange_factory(
        roundtrip_id=repository.roundtrip.id, sequence=2,
        compacted=True,
        call_overrides={"content": "search"},
        result_overrides={"payload": {"tool_name": "web_search", "evidence": [{"id": "ev-1"}]}},
    )
    repository.messages.extend([call, result])
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.kind for message in context.messages] == ["user", "assistant"]
    assert repository.list_messages(repository.roundtrip.id)[-2:] == [call, result]
    assert result.payload["evidence"] == [{"id": "ev-1"}]


@pytest.mark.parametrize("memory_payload", [
    {"operation": "working_memory_replaced"},
    {"error": "write failed"},
])
def test_context_does_not_reconstruct_compaction_from_memory_results(
    conversation_message_factory,
    conversation_tool_exchange_factory,
    context_repository_factory,
    memory_payload,
):
    repository = context_repository_factory()
    call, result = conversation_tool_exchange_factory(
        roundtrip_id=repository.roundtrip.id, sequence=2, call_overrides={"content": "search"},
    )
    memory = conversation_message_factory(
        roundtrip_id=repository.roundtrip.id,
        sequence=4,
        kind="tool_result",
        visibility="internal",
        role="tool",
        content="",
        payload={"tool_name": "update_memory_state", "output_payload": {"result": memory_payload}},
    )
    repository.messages.extend([call, result, memory])
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.id for message in context.messages] == [message.id for message in repository.messages]


def test_context_filters_compaction_independently_across_roundtrips(
    conversation_message_factory,
    context_repository_factory,
):
    repository = context_repository_factory()
    second = replace(repository.roundtrip, id=uuid4(), message_index=2)
    repository = context_repository_factory(
        roundtrips=[repository.roundtrip, second], messages=repository.messages,
    )
    compacted = conversation_message_factory(
        roundtrip_id=repository.roundtrip.id,
        sequence=2,
        kind="tool_result",
        visibility="internal",
        role="tool",
        content="",
        payload={"tool_name": "web_search"},
        compacted=True,
    )
    later = replace(compacted, id=uuid4(), roundtrip_id=second.id, compacted=False)
    repository.messages.extend([compacted, later])
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert compacted.id not in {message.id for message in context.messages}
    assert later.id in {message.id for message in context.messages}


def test_context_projects_out_internal_trace_messages(
    conversation_message_factory,
    context_repository_factory,
):
    repository = context_repository_factory()
    repository.messages.insert(1, conversation_message_factory(
        roundtrip_id=repository.roundtrip.id,
        sequence=1,
        kind="assistant",
        visibility="internal",
        role="assistant",
        content="full prompt should not be context",
        payload={"sections_raw": {"large": "x" * 10000}},
    ))
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert all(message.visibility == "visible" for message in context.messages)


def test_context_excludes_subagent_tool_entries(conversation_message_factory, context_repository_factory):
    repository = context_repository_factory()
    repository.messages.append(conversation_message_factory(
        roundtrip_id=repository.roundtrip.id,
        sequence=2,
        kind="tool_result",
        visibility="internal",
        role="tool",
        content="",
        payload={"tool_name": "web_search"},
        is_subagent=True,
        subagent_name="delegate",
    ))
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.kind for message in context.messages] == ["user", "assistant"]
