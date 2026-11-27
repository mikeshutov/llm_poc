from unittest.mock import patch

from conversation.context_builder import build_roundtrip_context


def test_context_uses_ordered_visible_messages(context_repository_factory) -> None:
    repository = context_repository_factory()
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.content for message in context.messages] == ["Find boots.", "Here are boots."]
    assert context.conversation_summary == ""


def test_context_projects_out_internal_trace_messages(conversation_message_factory, context_repository_factory) -> None:
    repository = context_repository_factory()
    repository.messages.insert(1, conversation_message_factory(roundtrip_id=repository.roundtrip.id, sequence=1, kind="assistant", visibility="internal", role="assistant", content="full prompt should not be context", payload={"sections_raw": {"large": "x" * 10000}}))
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert all(message.visibility == "visible" for message in context.messages)


def test_context_excludes_subagent_tool_entries(conversation_message_factory, context_repository_factory) -> None:
    repository = context_repository_factory()
    repository.messages.extend([
        conversation_message_factory(roundtrip_id=repository.roundtrip.id, sequence=2, kind="tool_call", visibility="internal", role="assistant", content="document_writing: create the report", payload={"tool_name": "document_writing"}, metadata={"tool_name": "document_writing"}, is_subagent=True, subagent_name="document_writing"),
        conversation_message_factory(roundtrip_id=repository.roundtrip.id, sequence=3, kind="tool_result", visibility="internal", role="tool", payload={"tool_name": "document_writing", "output_payload": {"task": "create the report"}}, metadata={"tool_name": "document_writing"}, is_subagent=True, subagent_name="document_writing"),
    ])
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert all(message.is_subagent is not True for message in context.messages)
