from uuid import uuid4

from conversation.models.conversation_models import ConversationRoundtrip
from tool.formatting import build_roundtrip_messages


def test_summary_messages_include_relevant_evidence_by_tool() -> None:
    evidence_id = uuid4()
    roundtrip = ConversationRoundtrip(
        id=uuid4(),
        conversation_id=uuid4(),
        message_index=0,
        user_prompt="Find a red shirt.",
        generated_response="Here are some red shirts.",
        roundtrip_summary=None,
        roundtrip_summary_embedding=None,
        response_payload={},
        parsed_query={},
        created_at="2026-08-27T00:00:00Z",
        metadata={},
        relevant_evidence={"search_products": [evidence_id]},
    )

    messages = build_roundtrip_messages([roundtrip])

    assert messages[1]["content"] == (
        "Here are some red shirts.\n"
        "Relevant Evidence by Tool:\n"
        f'{{"search_products":["{evidence_id}"]}}'
    )
