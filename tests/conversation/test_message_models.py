from uuid import uuid4

from conversation.models.conversation_models import (
    ConversationMessage,
    ConversationMessageKind,
    ConversationMessageVisibility,
)


def test_conversation_message_normalizes_payload_and_metadata() -> None:
    message = ConversationMessage(
        id=uuid4(),
        roundtrip_id=uuid4(),
        sequence=2,
        kind=ConversationMessageKind.TOOL_RESULT,
        visibility=ConversationMessageVisibility.INTERNAL,
        role="tool",
        content="",
        payload={"result": {"ok": True}},
        metadata={"latency_ms": 12},
        parent_message_id=None,
        created_at="now",
    )

    assert message.kind == "tool_result"
    assert message.visibility == "internal"
    assert message.payload["result"]["ok"] is True
    assert message.metadata["latency_ms"] == 12
