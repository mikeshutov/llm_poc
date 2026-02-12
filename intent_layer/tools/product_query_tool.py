from intent_layer.tools.query_intent_tool import PRODUCT_QUERY_DESCRIPTION

# Tool used to extract uncommon product-specific properties
PRODUCT_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_product_query",
        "description": PRODUCT_QUERY_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "product_query": {
                    "type": "object",
                    "properties": {
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
                    },
                }
            },
            "required": ["product_query"],
            "additionalProperties": False,
        },
    },
}
