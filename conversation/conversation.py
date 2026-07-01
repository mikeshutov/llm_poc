from uuid import UUID

from llm.clients.llm_client import LlmClient
from conversation.prompts.title_prompt import SYSTEM_PROMPT
from conversation.prompts.summary_prompt import build_prompt as build_summary_prompt
from conversation.models.conversation_models import ConversationRoundtrip, ConversationSummaryResponse
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER
from common.parsing import strip_code_fences
from conversation.utils import flatten_conversation_entries
from tool.repository.models import ToolCall
from tool.formatting import build_roundtrip_messages

llm = LlmClient()

def generate_conversation_title(prompt: str) -> str:
    res = llm.call_with_tools(
        SYSTEM_PROMPT,
        [{ROLE_KEY: ROLE_USER, CONTENT_KEY: prompt}],
        tools=[],
    )
    title = (res.raw_message.content or "").strip()[:60]
    return title or " ".join(prompt.split()).strip()[:60] or "Untitled"


def generate_conversation_summary(
    roundtrips: list[ConversationRoundtrip],
    tool_call_map: dict[UUID, list[ToolCall]] | None = None,
    previous_summary: str | None = None,
) -> ConversationSummaryResponse:
    roundtrip_messages = build_roundtrip_messages(roundtrips, tool_call_map)
    messages = [{ROLE_KEY: ROLE_USER, CONTENT_KEY: flatten_conversation_entries(roundtrip_messages)}]
    result = llm.call_with_tools(
        build_summary_prompt(previous_summary),
        messages,
        tools=[],
    )
    raw = strip_code_fences(result.raw_message.content or "")
    try:
        return ConversationSummaryResponse.model_validate_json(raw)
    except Exception:
        return ConversationSummaryResponse(conversation_summary=raw)
