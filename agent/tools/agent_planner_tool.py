AGENT_PLANNER_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_agent_step",
        "description": "Determines whether the current search goal is complete and what to do next.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Concrete search goal for the current request.",
                },
                "done": {
                    "type": "boolean",
                    "description": "True when current retrieved results are enough to satisfy the goal.",
                },
                "weather_request": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "year": {"type": "integer"},
                        "month": {"type": "integer"},
                        "purpose": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "notes": {
                    "type": "string",
                    "description": "Optional short rationale for debugging.",
                },
            },
            "required": ["goal", "done"],
            "additionalProperties": False,
        },
    },
}

FINAL_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "final_decision",
        "description": "Terminal planner decision for the current iteration loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "done": {"type": "boolean"},
                "weather_request": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "year": {"type": "integer"},
                        "month": {"type": "integer"},
                        "purpose": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "query_refinement_reason": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["goal", "done"],
            "additionalProperties": False,
        },
    },
}

PRODUCT_CATEGORY_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "list_product_categories",
        "description": "Returns available product categories from the internal catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of categories to return. For PoC breadth, prefer 200.",
                    "default": 200,
                }
            },
            "additionalProperties": False,
        },
    },
}

PLANNER_FIND_PRODUCTS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_products",
        "description": "Search internal catalog first; runtime controls whether fallback web search is allowed. Do not invent or assume categories only use categories from other tools. Refine Query text to a two 3 word query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "common_filters": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                        "price_min": {"type": "number"},
                        "price_max": {"type": "number"},
                        "gender": {
                            "type": "string",
                            "enum": ["Men", "Women", "none", "non"],
                        },
                    },
                    "additionalProperties": False,
                },
                "product_filters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category to filter by. Use values returned by list_product_categories.",
                        },
                        "style": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "web_count": {"type": "integer"},
            },
            "required": ["query_text"],
            "additionalProperties": False,
        },
    },
}

PLANNER_RESOLVE_CITY_LOCATION_TOOL = {
    "type": "function",
    "function": {
        "name": "resolve_city_location",
        "description": "Resolve a city into normalized location metadata for weather-aware shopping decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

PLANNER_GET_HISTORICAL_MONTH_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_historical_month_weather",
        "description": "Fetch historical monthly weather summary for a city, month, and year.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "year": {"type": "integer"},
                "month": {"type": "integer"},
            },
            "required": ["city", "year", "month"],
            "additionalProperties": False,
        },
    },
}

PLANNER_GENERIC_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "generic_web_search",
        "description": "Run web or news search for general information requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "search_type": {
                    "type": "string",
                    "enum": ["web_search", "news_search", "suggestion_search"],
                },
            },
            "required": ["query_text"],
            "additionalProperties": False,
        },
    },
}
