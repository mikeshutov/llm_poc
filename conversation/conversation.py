from llm.clients.llm_client import LlmClient
from conversation.prompts.title_prompt import SYSTEM_PROMPT
from conversation.prompts.summary_prompt import SYSTEM_PROMPT as SUMMARY_SYSTEM_PROMPT
from conversation.models.conversation_models import ConversationRoundtrip
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM, ROLE_USER

llm = LlmClient()

def generate_conversation_title(prompt: str) -> str:
    res = llm.call_with_tools(
        SYSTEM_PROMPT,
        [{ROLE_KEY: ROLE_USER, CONTENT_KEY: prompt}],
        tools=[],
        temperature=0, #experiment with 0
    )
    title = (res.raw_message.content or "").strip()[:60]
    return title or " ".join(prompt.split()).strip()[:60] or "Untitled"


def _format_roundtrips(roundtrips: list[ConversationRoundtrip]) -> str:
    lines: list[str] = []
    for rt in roundtrips:
        lines.append(f"User: {rt.user_prompt}")
        lines.append(f"Assistant: {rt.generated_response}")
    return "\n".join(lines)


def generate_conversation_summary(
    previous_summary: str | None,
    roundtrips: list[ConversationRoundtrip],
) -> str:
    messages: list[dict] = []
    if previous_summary:
        messages.append(
            {
                ROLE_KEY: ROLE_SYSTEM,
                CONTENT_KEY: (
                    "Low-priority context (may be outdated):\n"
                    f"{previous_summary}"
                ),
            }
        )
    messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: _format_roundtrips(roundtrips)})
    result = llm.call_with_tools(
        SUMMARY_SYSTEM_PROMPT,
        messages,
        tools=[],
        temperature=0.2,
    )
    return (result.raw_message.content or "").strip()
