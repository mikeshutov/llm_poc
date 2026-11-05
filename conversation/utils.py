import json

from common.data import prune_empty_prompt_values, sanitize_for_json_storage
from conversation.models.conversation_models import ConversationContext


def flatten_conversation_entries(entries: list[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in entries
        if m.get("content")
    )


def build_conversation_context_json(context: ConversationContext) -> str:
    json_safe_context = sanitize_for_json_storage(
        prune_empty_prompt_values(context.model_dump(mode="json"))
    )
    return json.dumps(
        json_safe_context,
        indent=2,
        ensure_ascii=True,
        default=str,
    )
