from typing import Any, Dict
from uuid import UUID

from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM
from conversation.repository.repo_factory import get_conversation_repo
from llm.clients.llm_client import LlmClient
from personalization.tone import resolve_conversation_tone_label
from response_layer.prompts.response_generation_prompt import RESPONSE_SYSTEM_PROMPT
from response_layer.prompts.response_tones import (
    FRIENDLY_TONE_PROMPT,
    NEUTRAL_TONE_PROMPT,
    PROFESSIONAL_TONE_PROMPT,
)
from personalization.tone.models import ToneLabel
from response_layer.models.response_payload import ResponsePayload
from response_layer.tools.response_tool import RESPONSE_TOOL

# payload still dict, not ideal but its fine for this.
def _convert_to_response_payload(payload: Dict[str, Any] | None, fallback_text: str = "Something went wrong converting to response.") -> ResponsePayload:
    if payload is None:
        return ResponsePayload(
            response=fallback_text,
            cards=[],
            follow_up="",
        )
    response = str(payload.get("response") or "")
    cards = payload.get("cards")
    if cards is None:
        cards = payload.get("products")
    cards = cards or []
    follow_up = str(payload.get("follow_up") or "")
    if not isinstance(cards, list):
        cards = []
    return ResponsePayload(response=response, cards=cards, follow_up=follow_up)


def _tone_prompt(label: str | None) -> str:
    match label:
        case ToneLabel.FRIENDLY.value:
            return FRIENDLY_TONE_PROMPT
        case ToneLabel.PROFESSIONAL.value:
            return PROFESSIONAL_TONE_PROMPT
        case ToneLabel.NEUTRAL.value | None:
            return NEUTRAL_TONE_PROMPT
        case _:
            return NEUTRAL_TONE_PROMPT


def _resolve_tone_label(conversation_id: str | None) -> str | None:
    if not conversation_id:
        return None
    try:
        return resolve_conversation_tone_label(
            conversation_repository=get_conversation_repo(),
            conversation_id=UUID(conversation_id),
            parsed_tone=None,
        )
    except Exception:
        return None


def generate_response(
    conversation_entries: list[dict],
    query_results: str,
    conversation_id: str | None = None,
) -> ResponsePayload:
    llm = LlmClient()
    tone_label = _resolve_tone_label(conversation_id)
    messages = [*conversation_entries, {ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: query_results}]
    system_prompt = RESPONSE_SYSTEM_PROMPT + "\n" + _tone_prompt(tone_label)
    result = llm.call_with_tools(
        system_prompt,
        messages,
        tools=[RESPONSE_TOOL],
        temperature=0.2,
    )
    tool_call = result.tool_calls_by_name.get("build_response", [None])[-1]
    return _convert_to_response_payload(tool_call.args if tool_call else None)
