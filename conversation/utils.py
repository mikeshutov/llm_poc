def flatten_conversation_entries(entries: list[dict]) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in entries
        if m.get("content")
    )
