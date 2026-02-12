from intent_layer.models.intent import Intent
from websearch.models.search_type import SearchType
from personalization.tone.models import ToneLabel

QUERY_INTENT_DESCRIPTION = (
        "Determines user intent as well as any qualifying queries."
)

PRODUCT_QUERY_DESCRIPTION = (
        "Extracts structured filters from a user's product search query. "
        "Use null or omit a field if the user did not specify it. "
        "Price fields are numbers, interpreted as the user's currency."
)

# Tool used to determine user intent and common parameters
QUERY_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_query",
        "description": QUERY_INTENT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "User's likely intent based on provided enum.",
                    "enum": [i.value for i in Intent],
                },
                "tone": {
                    "type": "object",
                    "description": "Desired response tone inferred from the user's prompt.",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": [t.value for t in ToneLabel],
                            "description": "Tone label.",
                        },
                        "score": {
                            "type": "number",
                            "description": "Confidence score from 0 to 1.",
                        },
                        "override": {
                            "type": "boolean",
                            "description": "True if tone should switch due to a rapid change in conversation.",
                        },
                    },
                    "required": ["label", "score", "override"],
                },
                "common_properties": {
                    "type": "object",
                    "description": "Common properties reusable across intents.",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "Color filter when applicable.",
                        },
                        "price_min": {
                            "type": "number",
                            "description": "Minimum price if specified.",
                        },
                        "price_max": {
                            "type": "number",
                            "description": "Maximum price if specified.",
                        },
                        "gender": {
                            "type": "string",
                            "description": "Gender filter when applicable. Valid options are (men, women).",
                        },
                    },
                },
                "query_details": {
                    "type": "object",
                    "description": "Common query metadata usable across intents.",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Normalized query text capturing the user's core ask.",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key terms extracted from the user's request.",
                        },
                        "language": {
                            "type": "string",
                            "description": "ISO language code for the user's request, e.g. 'en', 'fr'.",
                        },
                        "search_type": {
                            "type": "string",
                            "description": "Search modality for general information requests.",
                            "enum": [s.value for s in SearchType],
                        },
                    },
                },
            },
            "required": ["intent", "tone"],
            "additionalProperties": False,
        },
    },
}
