from __future__ import annotations

from conversation.models.conversation_models import ConversationContext


def build_agent_selection_query(
    *,
    current_user_request: str,
    conversation_context: ConversationContext,
) -> str:
    sections: list[str] = []
    visible_messages = [
        message
        for message in conversation_context.messages
        if str(message.visibility) == "visible"
    ]
    previous_user_request = next(
        (
            message.content.strip()
            for message in reversed(visible_messages)
            if str(message.kind) == "user" and message.content.strip()
        ),
        "",
    )
    if previous_user_request:
        sections.append(f"Previous user request:\n{previous_user_request}")

    assistant_follow_up = next(
        (
            str(message.payload.get("next_question", "")).strip()
            for message in reversed(visible_messages)
            if str(message.kind) in {"assistant", "synthesis"}
            and isinstance(message.payload, dict)
            and str(message.payload.get("next_question", "")).strip()
        ),
        "",
    )
    if assistant_follow_up:
        sections.append(f"Latest assistant follow-up:\n{assistant_follow_up}")

    sections.append(f"Current user request:\n{current_user_request.strip()}")
    return "\n\n".join(sections)
