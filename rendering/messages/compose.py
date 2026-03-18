from uuid import UUID

from agent.tool.repository.models import ToolCall
from agent.tool.repository.tool_call_repository import ToolCallRepository
from common.message_constants import CONTENT_KEY, ROLE_ASSISTANT, ROLE_KEY, ROLE_USER
from conversation.models.conversation_models import ConversationRoundtrip

#compose messages from roundtrips including tool calls
def compose_messages_from_roundtrips(
    roundtrips: list[ConversationRoundtrip],
):
    roundtrip_ids = [rt.id for rt in roundtrips]
    tool_calls_by_roundtrip: dict[UUID, list[ToolCall]] = (
        ToolCallRepository().get_tool_calls_by_roundtrips(roundtrip_ids)
        if roundtrip_ids else {}
    )

    messages = []
    for rt in roundtrips:
        messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt})
        messages.append({ROLE_KEY: ROLE_ASSISTANT, CONTENT_KEY: _system_response_with_tools(rt.generated_response, tool_calls_by_roundtrip.get(rt.id, []))})
    return messages


def _system_response_with_tools(response: str, related_tools : list[ToolCall]):
    if not related_tools:
        return response
    
    tool_calls_or_summaries = []
    for tc in related_tools:
        tool_calls_or_summaries.append(tc.summary if tc.summary else f"input={tc.input_payload} output={tc.output_payload}")
    return f"{response}\nPrevious Tool Calls:\n" + "\n".join(tool_calls_or_summaries)
