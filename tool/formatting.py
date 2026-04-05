from uuid import UUID

from tool.repository.models import ToolCall
from conversation.models.conversation_models import ConversationRoundtrip
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER, ROLE_ASSISTANT


def format_tool_calls(response: str, tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return response
    # search_file_for_details returns a fairly large of content. For now just going to exclude but a better option is possible likely.
    CONTEXT_EXCLUDED_TOOLS = {"search_file_for_details"}
    summaries = [
        tc.summary if tc.summary else f"input={tc.input_payload} output={tc.output_payload}"
        for tc in tool_calls
        if tc.tool_name not in CONTEXT_EXCLUDED_TOOLS
    ]
    return f"{response}\nPrevious Tool Calls:\n" + "\n".join(summaries)


def build_roundtrip_messages(
    roundtrips: list[ConversationRoundtrip],
    tool_calls_by_roundtrip: dict[UUID, list[ToolCall]] | None = None,
) -> list[dict]:
    messages = []
    for rt in roundtrips:
        tool_calls = (tool_calls_by_roundtrip or {}).get(rt.id, [])
        messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: rt.user_prompt})
        messages.append({ROLE_KEY: ROLE_ASSISTANT, CONTENT_KEY: format_tool_calls(rt.generated_response, tool_calls)})
    return messages
