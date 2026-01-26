# Tool used to determine user intent
QUERY_INTENT_DESCRIPTION = "Generate a conversation title based on the user's prompt."

CONVERSATION_NAME_TOOL = {
    "type": "function",
    "function": {
        "name": "conversation_name",
        "description": QUERY_INTENT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A concise conversation title (max 40 chars) from the user's prompt.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}