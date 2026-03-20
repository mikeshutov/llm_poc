from tool.repository.tool_call_repository import ToolCallRepository
from tool.formatting import build_roundtrip_messages
from conversation.models.conversation_models import ConversationRoundtrip

#compose messages from roundtrips including tool calls
def compose_messages_from_roundtrips(
    roundtrips: list[ConversationRoundtrip],
):
    roundtrip_ids = [rt.id for rt in roundtrips]
    tool_calls_by_roundtrip = (
        ToolCallRepository().get_tool_calls_by_roundtrips(roundtrip_ids)
        if roundtrip_ids else {}
    )
    return build_roundtrip_messages(roundtrips, tool_calls_by_roundtrip)
