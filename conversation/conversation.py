from llm.clients.llm_client import get_llm_client
from conversation.prompts.title_prompt import SYSTEM_PROMPT
from conversation.prompts.summary_prompt import build_prompt as build_summary_prompt
from conversation.models.conversation_models import ConversationRoundtrip, ConversationSummaryResponse
from common.config import CONTENT_KEY, ROLE_ASSISTANT, ROLE_KEY, ROLE_USER
from common.data import strip_code_fences
from conversation.utils import flatten_conversation_entries
from conversation.repository.repo_factory import get_conversation_repo


def generate_conversation_title(prompt: str) -> str:
    res = get_llm_client().call_with_tools(
        SYSTEM_PROMPT,
        [{ROLE_KEY: ROLE_USER, CONTENT_KEY: prompt}],
        tools=[],
    )
    title = (res.raw_message.content or "").strip()[:60]
    return title or " ".join(prompt.split()).strip()[:60] or "Untitled"


def generate_conversation_summary(
    roundtrips: list[ConversationRoundtrip],
    previous_summary: str | None = None,
) -> ConversationSummaryResponse:
    repository = get_conversation_repo()
    entries = []
    for roundtrip in roundtrips:
        entries.extend(
            {ROLE_KEY: message.role, CONTENT_KEY: message.content}
            for message in repository.list_messages(roundtrip.id, include_internal=False)
            if message.role in {ROLE_USER, ROLE_ASSISTANT} and message.content.strip()
        )
    result = get_llm_client().call_with_tools(
        build_summary_prompt(previous_summary),
        [
            {
                ROLE_KEY: ROLE_USER,
                CONTENT_KEY: flatten_conversation_entries(entries),
            }
        ],
        tools=[],
    )
    raw = strip_code_fences(result.raw_message.content or "")
    try:
        return ConversationSummaryResponse.model_validate_json(raw)
    except Exception:
        return ConversationSummaryResponse(conversation_summary=raw)
