
from conversation.models.conversation_models import ConversationContext
from agents.discovery.agent_selection_query import build_agent_selection_query


def test_build_agent_selection_query_includes_turn_continuity(conversation_message_factory) -> None:
    query = build_agent_selection_query(
        current_user_request="Find a less expensive option.",
        conversation_context=ConversationContext(
            messages=[
                conversation_message_factory(content="Find me a hiking backpack."),
                conversation_message_factory(
                    sequence=1,
                    kind="synthesis",
                    role="assistant",
                    content="Here are some options.",
                    payload={"next_question": "What budget should I use?"},
                ),
            ],
        ),
    )

    assert query == (
        "Previous user request:\nFind me a hiking backpack.\n\n"
        "Latest assistant follow-up:\nWhat budget should I use?\n\n"
        "Current user request:\nFind a less expensive option."
    )
