import json


def flatten_conversation_entries(entries: list[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in entries
        if m.get("content")
    )


from conversation.models.conversation_models import ConversationContext


def build_conversation_context_json(context: ConversationContext) -> str:
    return json.dumps(context.model_dump(), indent=2, ensure_ascii=True)
