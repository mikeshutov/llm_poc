RESPONSE_TOOL = {
    "type": "function",
    "function": {
        "name": "build_response",
        "description": "Builds a structured response with cards and a follow-up.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "cards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Stable identifier for the item (database id or URL-based id).",
                            },
                            "name": {
                                "type": "string",
                                "description": "Display name or title of the item.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Short summary or snippet suitable for a card.",
                            },
                            "price": {
                                "type": ["number", "string"],
                                "description": "Price value if available; may be numeric or a formatted string.",
                            },
                            "url": {
                                "type": "string",
                                "description": "Destination URL for the item; may be empty if unavailable.",
                            },
                            "image_url": {
                                "type": "string",
                                "description": "Image URL for the card thumbnail; may be empty if unavailable.",
                            },
                            "source": {
                                "type": "string",
                                "description": "Origin of the item, e.g. 'db' or 'web'.",
                            },
                        },
                        "required": ["id", "name"],
                        "additionalProperties": False,
                    },
                },
                "follow_up": {"type": "string"},
            },
            "required": ["response", "cards", "follow_up"],
            "additionalProperties": False,
        },
    },
}
