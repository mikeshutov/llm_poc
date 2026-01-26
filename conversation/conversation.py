from llm.clients.llm_client import LlmClient
from intent_layer.tools.conversation_name_tool import CONVERSATION_NAME_TOOL
from prompts.title_prompt import SYSTEM_PROMPT

llm = LlmClient()

# generate a title from the provided prompt used for initial conversation setup
def generate_conversation_title(prompt: str) -> str:
    res = llm.call_with_tools(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], [CONVERSATION_NAME_TOOL], temperature=0.2)
    # this might be later moved into a fucntion if we decide that tools can be orchestrated differently.
    title_call = res.tool_calls_by_name.get("conversation_name", [None])[-1]
    if title_call and title_call.args and title_call.args.get("title"):
        return title_call.args["title"].strip()[:60]
    return " ".join(prompt.split()).strip()[:60] or "Untitled"