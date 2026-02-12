from llm.clients.llm_client import LlmClient
from intent_layer.tools.conversation_name_tool import CONVERSATION_NAME_TOOL
from conversation.prompts.title_prompt import SYSTEM_PROMPT
from conversation.prompts.summary_prompt import SYSTEM_PROMPT as SUMMARY_SYSTEM_PROMPT
from conversation.models.conversation_models import ConversationRoundtrip

llm = LlmClient()

# generate a title from the provided prompt used for initial conversation setup
def generate_conversation_title(prompt: str) -> str:
    res = llm.call_with_tools(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], [CONVERSATION_NAME_TOOL], temperature=0.2)
    # this might be later moved into a fucntion if we decide that tools can be orchestrated differently.
    title_call = res.tool_calls_by_name.get("conversation_name", [None])[-1]
    if title_call and title_call.args and title_call.args.get("title"):
        return title_call.args["title"].strip()[:60]
    return " ".join(prompt.split()).strip()[:60] or "Untitled"


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
                "role": "system",
                "content": (
                    "Low-priority context (may be outdated):\n"
                    f"{previous_summary}"
                ),
            }
        )
    messages.append({"role": "user", "content": _format_roundtrips(roundtrips)})
    result = llm.call_with_tools(
        SUMMARY_SYSTEM_PROMPT,
        messages,
        tools=[],
        temperature=0.2,
    )
    return (result.raw_message.content or "").strip()
