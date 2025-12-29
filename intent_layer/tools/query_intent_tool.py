from models.intent import Intent

QUERY_INTENT_DESCRIPTION = (
        "Determines user intent as well as any qualifying queries."
        "Current intent valid values are find_products or unknown."
)

PRODUCT_QUERY_DESCRIPTION = (
        "Extracts structured filters from a user's product search query. "
        "Use null or omit a field if the user did not specify it. "
        "Price fields are numbers, interpreted as the user's currency."
)

# Tool used to determine user intent
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
                    "description": "User intent, usually 'find_products' or 'unknown'.",
                    "enum": [i.value for i in Intent],
                },
                "product_query":{
                    "description": PRODUCT_QUERY_DESCRIPTION,
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "Free-form semantic description of what the user is looking for to be used for embeddings",
                        },
                        "category": {
                            "type": "string",
                            "description": "Product category like 'shirt', 'shoes', 'pants'.",
                        },
                        "color": {
                            "type": "string",
                            "description": "Color mentioned by the user, e.g. 'red', 'blue'.",
                        },
                        "price_min": {
                            "type": "number",
                            "description": "Minimal price if the user mentions something like 'over 50'.",
                        },
                        "price_max": {
                            "type": "number",
                            "description": "Max price if user mentions 'under 50', 'less than 100', etc.",
                        },
                        "style": {
                            "type": "string",
                            "description": "Style or context, e.g. 'gym', 'formal', 'casual'.",
                        },
                        "size_label": {
                            "type": "string",
                            "description": "Size label like 'S', 'M', 'L', 'XL', 'XXL', 'small', 'medium', 'large'. Use this only if the user explicitly uses a label."
                        },
                        "size_numeric": {
                            "type": "number",
                            "description": "Numeric size if the user mentions measurements like '16.5 inch', '42 chest', '32 waist'."
                        },
                        "size_unit": {
                            "type": "string",
                            "description": "Unit of the numeric size (e.g. 'inch', 'cm', 'waist', 'neck', 'chest')."
                        },
                        "gender": {
                            "type": "string",
                            "description": "The gender of the user or product. Valid options are (men, women)"
                        },
                    }
                }
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}