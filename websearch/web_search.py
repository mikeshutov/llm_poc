
from pydantic import ValidationError

from intent_layer.tools.general_search_tool import GENERAL_SEARCH_TOOL
from llm.clients.llm_client import LlmClient
from models.general_search_query import GeneralSearchQuery
from models.search_type import SearchType
from prompts.search_prompt import SYSTEM_PROMPT

llm = LlmClient()

# generate search tool results to be used to perform a web search based on input
def process_search_intent(prompt: str) -> GeneralSearchQuery:
    tool_result =  llm.call_with_tools(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], [GENERAL_SEARCH_TOOL], temperature=0.2)
    intent_call = tool_result.tool_calls_by_name.get("general_search_processing_tool", [None])[-1]
    if intent_call is None or not intent_call.args:
        return GeneralSearchQuery(search_query="",search_type=SearchType.WEB_SEARCH)

    try:
        return GeneralSearchQuery(**intent_call.args)
    except ValidationError:
        return GeneralSearchQuery(search_query="",search_type=SearchType.WEB_SEARCH)
