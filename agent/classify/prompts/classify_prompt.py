from agent.agentstate.model import AgentState, flatten_conversation_entries
from agent.classify.prompts.classify_schema_prompt import CLASSIFY_SCHEMA
from agent.tool.tools import TOOL_CATEGORIES


def build_classify_prompt(agent_state: AgentState) -> str:
    conversation_block = ""
    if agent_state.conversation_entries:
        flat = flatten_conversation_entries(agent_state.conversation_entries)
        conversation_block = f"Conversation history:\n{flat}\n\n"

    category_lines = "\n".join(
        f"- Category: {name} | Category Description: {cat.description}"
        for name, cat in TOOL_CATEGORIES.items()
    )

    return (
        "You are a request classifier. Given the user's request, return only the category names that are relevant.\n\n"
        f"{conversation_block}"
        f"User request: {agent_state.task}\n\n"
        f"Available categories:\n{category_lines}\n\n"
        f"Response Schema: {CLASSIFY_SCHEMA}"
    )