from uuid import UUID

from tool.repository.models import ToolCall
from tool.repository.tool_call_repository import ToolCallRepository
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER
from llm.clients.llm_client import LlmClient

# This is just a start but we need more dynamic prompts so that tools 
# that require more context maintain it. As it stands we lose too much context
SYSTEM_PROMPT = "Summarize the following tool call in one concise sentence. Focus on what was looked up and what the result was."

llm = LlmClient()

def summarize_tool_calls(roundtrip_id: UUID) -> None:
    tool_repo = ToolCallRepository()
    tool_calls = tool_repo.get_tool_calls_by_roundtrips([roundtrip_id]).get(roundtrip_id, [])
    for tc in tool_calls:
        if tc.summary:
            continue
        summary = _summarize_tool_call(tc)
        tool_repo.update_tool_call_summary(tc.id, summary)

def _summarize_tool_call(tc: ToolCall) -> str:
    prompt = (
        f"Tool: {tc.tool_name}\n"
        f"Input: {tc.input_payload}\n"
        f"Output: {tc.output_payload}"
    )
    res = llm.call_with_tools(
        SYSTEM_PROMPT,
        [{ROLE_KEY: ROLE_USER, CONTENT_KEY: prompt}],
        tools=[],
        temperature=0,
    )
    return (res.raw_message.content or "").strip()