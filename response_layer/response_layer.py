from openai import OpenAI
from prompts.response_generation_prompt import RESPONSE_SYSTEM_PROMPT

client = OpenAI()
def generate_response(parsedRequest: list[dict], query_results):
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
            *parsedRequest,
            {"role": "system", "content": query_results},
        ],
    )


    return completion.choices[0].message.content or ""