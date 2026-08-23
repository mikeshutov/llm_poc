from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from integrations.free_dictionary import FreeDictionaryClient
from integrations.free_dictionary.models import DictionaryEntry
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView, ToolResult
from tool.constants import TOOL_NAME_DEFINE_WORD
from tool.constants import TOOL_RESULT_TYPE_DEFINITION

_dictionary_client = FreeDictionaryClient()


class DefineWordArgs(BaseModel):
    word: str = Field(
        ...,
        description="The word to look up in the dictionary.",
    )


class DictionaryEntryMetadata(BaseModel):
    phonetic: str | None = None
    meaning_count: int


def _entry_summary(entry: DictionaryEntry) -> str:
    if entry.meanings and entry.meanings[0].definitions:
        return entry.meanings[0].definitions[0].definition
    return f"Dictionary entry for {entry.word}."


def _tool_result(result: list[DictionaryEntry]) -> ToolResult:
    evidence: list[EvidenceView] = []
    for entry in result:
        source_url = entry.source_urls[0].strip() if entry.source_urls else ""
        metadata = DictionaryEntryMetadata(
            phonetic=entry.phonetic,
            meaning_count=len(entry.meanings),
        )
        evidence_view = EvidenceView(
            item_id=entry.word.strip(),
            tool_name=TOOL_NAME_DEFINE_WORD,
            title=entry.word.strip(),
            summary=_entry_summary(entry),
            urls=[EvidenceUrl(url=source_url, url_type=EvidenceUrlType.WEBSITE)] if source_url else [],
            source=TOOL_NAME_DEFINE_WORD,
            entity_type=TOOL_RESULT_TYPE_DEFINITION,
            llm_metadata=metadata.model_dump(exclude_none=True),
            raw_payload=entry,
        )
        evidence.append(evidence_view)
    return ToolResult(result=result, evidence=evidence)




@tool(
    TOOL_NAME_DEFINE_WORD,
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
def define_word(word: str) -> ToolResult:
    try:
        return _tool_result(_dictionary_client.define(word))
    except RequestException as e:
        return ToolResult.error(f"Dictionary service unavailable: {e}")
