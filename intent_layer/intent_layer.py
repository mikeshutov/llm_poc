import os, json
from openai import OpenAI
from models.parsed_request import ParsedRequest
from models.intent import Intent
from pydantic import ValidationError
from prompts.query_parser_prompt import SYSTEM_PROMPT
from intent_layer.tools.query_intent_tool import QUERY_INTENT_TOOL

client = OpenAI()

def parse_query(query: str) -> ParsedRequest:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        tools=[QUERY_INTENT_TOOL],
        temperature=0,
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return ParsedRequest(intent=Intent.UNKNOWN)

    raw_args = msg.tool_calls[0].function.arguments

    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception as e:
            return ParsedRequest(intent=Intent.UNKNOWN)

    try:
        parsed_request = ParsedRequest(**raw_args)
    except ValidationError as e:
        parsed_request = ParsedRequest(intent=Intent.UNKNOWN)

    return parsed_request