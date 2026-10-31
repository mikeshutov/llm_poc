from uuid import uuid4

from conversation.models.conversation_models import ConversationContext, ConversationMessage
from agents.discovery.agent_selection_query import build_agent_selection_query


def test_build_agent_selection_query_includes_turn_continuity() -> None:
    query = build_agent_selection_query(
        current_user_request="Find a less expensive option.",
        conversation_context=ConversationContext(
            messages=[
                ConversationMessage(
                    uuid4(), uuid4(), 0, "user", "visible", "user",
                    "Find me a hiking backpack.", {}, {}, None, "",
                ),
                ConversationMessage(
                    uuid4(), uuid4(), 1, "synthesis", "visible", "assistant",
                    "Here are some options.", {"next_question": "What budget should I use?"}, {}, None, "",
                ),
            ],
        ),
    )

    assert query == (
        "Previous user request:\nFind me a hiking backpack.\n\n"
        "Latest assistant follow-up:\nWhat budget should I use?\n\n"
        "Current user request:\nFind a less expensive option."
    )
