from conversation.models.conversation_models import ConversationContext
from request_orchestrator.shared.agents.agent_selection_query import build_agent_selection_query


def test_build_agent_selection_query_includes_turn_continuity() -> None:
    query = build_agent_selection_query(
        current_user_request="Find a less expensive option.",
        conversation_context=ConversationContext(
            previous_user_request="Find me a hiking backpack.",
            latest_assistant_follow_up="What budget should I use?",
        ),
    )

    assert query == (
        "Previous user request:\nFind me a hiking backpack.\n\n"
        "Latest assistant follow-up:\nWhat budget should I use?\n\n"
        "Current user request:\nFind a less expensive option."
    )
