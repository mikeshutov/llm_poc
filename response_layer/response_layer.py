import json
from openai import OpenAI

from models.parsed_request import ParsedRequest
from models.intent import Intent
from prompts.response_generation_prompt import RESPONSE_SYSTEM_PROMPT

client = OpenAI()


def generate_response(user_query: str, parsedRequest: ParsedRequest, query_results):
    """
    Keep this for now
    """
    base_context = {
        "user_query": user_query,
        "parsed_request": parsedRequest.model_dump(),
        "results_text": query_results,
    }
    print(base_context)

    if parsedRequest.intent == Intent.FIND_PRODUCTS:
        user_message = (
            f"The user asked:\n{user_query}\n\n"
            "Parsed request:\n"
            f"{json.dumps(parsedRequest.model_dump(), indent=2)}\n\n"
            "Here are the retrieved product results:\n"
            f"{query_results}\n\n"
            "Explain the best matching products to the user."
        )
    else:
        user_message = (
            f"The user asked:\n{user_query}\n\n"
            "Intent was not recognized.\n"
            "Help user refine their request."
        )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return completion.choices[0].message.content or ""