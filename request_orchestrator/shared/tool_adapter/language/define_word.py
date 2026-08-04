from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.free_dictionary import FreeDictionaryClient, DictionaryEntry

_dictionary_client = FreeDictionaryClient()


class DefineWordArgs(BaseModel):
    word: str = Field(
        ...,
        description="The word to look up in the dictionary.",
    )


@tool(
    "define_word",
    args_schema=DefineWordArgs,
    description="""
Look up the definition of an English word, including meanings, parts of speech, examples, synonyms, and antonyms.

Required fields:
- word (string)

Example valid call:
{
  "word": "serendipity"
}
""",
)
def define_word(word: str) -> list[DictionaryEntry] | str:
    try:
        return _dictionary_client.define(word)
    except RequestException as e:
        return f"Dictionary service unavailable: {e}"
