from __future__ import annotations

from conversation.models.conversation_models import ConversationContext


def build_agent_selection_query(
    *,
    current_user_request: str,
    conversation_context: ConversationContext,
) -> str:
    sections: list[str] = []
    previous_user_request = conversation_context.previous_user_request.strip()
    if previous_user_request:
        sections.append(f"Previous user request:\n{previous_user_request}")

    assistant_follow_up = conversation_context.latest_assistant_follow_up.strip()
    if assistant_follow_up:
        sections.append(f"Latest assistant follow-up:\n{assistant_follow_up}")

    sections.append(f"Current user request:\n{current_user_request.strip()}")
    return "\n\n".join(sections)
