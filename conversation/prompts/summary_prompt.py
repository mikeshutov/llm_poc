from conversation.prompts.summary_schema_prompt import SUMMARY_SCHEMA_PROMPT

SYSTEM_PROMPT = """
You summarize conversations for an assistant that works on multiple topics.

Rules:
- Ensure the topics discussed are included as well as a brief description of what was discussed.
- If the user has some sort of implied intent make sure to preserve that intent.
- Preserve any UUIDs or identifier referenced in the conversation along with the name and description of the entity referenced.
- When another summary is present it is considered lower priority unless relevant to newer mesages.
"""

def build_prompt() -> str:
    return f"{SYSTEM_PROMPT}\n\n{SUMMARY_SCHEMA_PROMPT}"
