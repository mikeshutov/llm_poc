from uuid import uuid4
from unittest.mock import patch

from conversation.context_builder import build_roundtrip_context
from conversation.models.conversation_models import Conversation, ConversationMessage, ConversationRoundtrip, ConversationSummary


class FakeConversationRepository:
    def __init__(self):
        self.roundtrip = ConversationRoundtrip(uuid4(), uuid4(), 1, "summary", None, "2026-01-01", {}, "completed")
        self.messages = [
            ConversationMessage(uuid4(), self.roundtrip.id, 0, "user", "visible", "user", "Find boots.", {}, {}, None, ""),
            ConversationMessage(uuid4(), self.roundtrip.id, 1, "assistant", "visible", "assistant", "Here are boots.", {"next_question": "What size?"}, {"stage": "synthesis"}, None, ""),
        ]

    def get_conversation(self, conversation_id):
        return Conversation(conversation_id, "user-1", "Test", "", {}, {}, "", None)

    def get_latest_summary(self, conversation_id):
        return ConversationSummary(uuid4(), conversation_id, "", 0, "")

    def list_roundtrips(self, *args, **kwargs):
        return [self.roundtrip]

    def list_messages(self, roundtrip_id, include_internal=True):
        return self.messages

def test_context_uses_ordered_visible_messages() -> None:
    repository = FakeConversationRepository()
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert [message.content for message in context.messages] == ["Find boots.", "Here are boots."]
    assert context.conversation_summary == ""


def test_context_projects_out_internal_trace_messages() -> None:
    repository = FakeConversationRepository()
    repository.messages.insert(1, ConversationMessage(
        uuid4(), repository.roundtrip.id, 1, "assistant", "internal", "assistant",
        "full prompt should not be context", {"sections_raw": {"large": "x" * 10000}}, {}, None, "",
    ))
    with patch("conversation.context_builder.get_conversation_repo", return_value=repository):
        context = build_roundtrip_context(str(repository.roundtrip.conversation_id))

    assert all(message.visibility == "visible" for message in context.messages)
