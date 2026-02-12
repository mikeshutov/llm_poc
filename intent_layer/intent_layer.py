from llm.clients.llm_client import LlmClient
from intent_layer.models.parsed_request import ParsedRequest
from intent_layer.models.intent import Intent
from pydantic import ValidationError
from intent_layer.prompts.query_parser_prompt import SYSTEM_PROMPT
from intent_layer.tools.query_intent_tool import QUERY_INTENT_TOOL

llm = LlmClient()

# parse_query is where we will figure out what tools we should add to orchestrate the call and get better results
def parse_query(messages: list[dict]) -> ParsedRequest:
    llm_parsed_query = llm.call_with_tools(SYSTEM_PROMPT, messages, [QUERY_INTENT_TOOL], temperature=0.0)

    intent_call = llm_parsed_query.tool_calls_by_name.get("parse_query", [None])[-1]
    if intent_call is None or not intent_call.args:
        return ParsedRequest(intent=Intent.UNKNOWN)

    try:
        parsed = ParsedRequest(**intent_call.args)
        return parsed
    except ValidationError:
        return ParsedRequest(intent=Intent.UNKNOWN)
