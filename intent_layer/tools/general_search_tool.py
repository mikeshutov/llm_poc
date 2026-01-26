#Tool used to get more details about the search we want to perform
from models.search_type import SearchType

QUERY_INTENT_DESCRIPTION = "Establish some information about what the user is looking for to provide more precise searches."

GENERAL_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "general_search_processing_tool",
        "description": QUERY_INTENT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "A concise conversation simplified search query based on the users input.",
                },
                "search_type": {
                    "type": "string",
                    "description": "The type of search to perform.",
                    "enum": [i.value for i in SearchType],
                },
            },
            "required": ["search_query", "search_type"],
            "additionalProperties": False,
        },
    },
}