from conversation.prompts.summary_schema_prompt import SUMMARY_SCHEMA_PROMPT

SYSTEM_PROMPT = """You summarize conversations for an assistant that works on multiple topics."""

def build_prompt() -> str:
    return f"{SYSTEM_PROMPT}\n\n{SUMMARY_SCHEMA_PROMPT}"
