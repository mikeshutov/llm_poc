from uuid import uuid4

from conversation.models.conversation_models import ConversationContext, ConversationMessage
from request_orchestrator.models.agent_prompt import AgentPrompt, PromptSectionKeys


def test_agent_prompt_builds_native_chat_messages() -> None:
    prompt = AgentPrompt(
        instruction="Answer helpfully.",
        rules="Use only the supplied evidence.",
        conversation_context=ConversationContext(
            conversation_summary="The user is shopping for boots.",
            messages=[
                ConversationMessage(uuid4(), uuid4(), 0, "user", "visible", "user", "I need waterproof boots.", {}, {}, None, ""),
                ConversationMessage(uuid4(), uuid4(), 1, "assistant", "visible", "assistant", "What size do you need?", {}, {}, None, ""),
            ],
        ),
        task="Find suitable options.",
    )
    prompt.include_section(PromptSectionKeys.RULES)
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)
    prompt.include_section(PromptSectionKeys.TASK)

    assert prompt.build_messages() == [
        {
            "role": "system",
            "content": (
                "Answer helpfully.\n\n"
                "Use only the supplied evidence.\n\n"
                "CONVERSATION SUMMARY\n"
                '{\n  "conversation_summary": "The user is shopping for boots."\n}'
            ),
        },
        {"role": "user", "content": "I need waterproof boots."},
        {"role": "assistant", "content": "What size do you need?"},
        {"role": "user", "content": "Find suitable options."},
    ]


def test_agent_prompt_sends_tool_evidence_instead_of_raw_output() -> None:
    prompt = AgentPrompt(
        instruction="",
        conversation_context=ConversationContext(
            messages=[
                ConversationMessage(
                    uuid4(), uuid4(), 0, "tool_result", "internal", "tool",
                        "ignored hypothesis", {
                        "tool_name": "search_products",
                        "status": "completed",
                        "output_payload": {"raw": "do not send this"},
                        "evidence": [{
                            "id": "evidence-1",
                            "title": "Boots",
                            "summary": "Waterproof boots",
                            "raw_payload": {"large": "raw response"},
                        }],
                    }, {}, None, "",
                ),
            ],
        ),
    )
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    messages = prompt.build_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "tool_call_id" not in messages[0]
    assert messages[0]["content"] == (
        '{\n  "tool_name": "search_products",\n  "status": "completed",\n'
        '  "error_message": null,\n  "duration_ms": null,\n'
        '  "evidence": [\n    {\n      "id": "evidence-1",\n'
        '      "title": "Boots",\n      "summary": "Waterproof boots"\n'
        '    }\n  ],\n  "tool_metadata": {}\n}'
    )


def test_agent_prompt_preserves_historical_tool_names() -> None:
    tool_call_id = uuid4()
    prompt = AgentPrompt(
        instruction="",
        conversation_context=ConversationContext(
            messages=[
                ConversationMessage(
                    tool_call_id, uuid4(), 0, "tool_call", "internal", "assistant", "", {
                        "tool_name": "products_search",
                        "provider_tool_call_id": "call-1",
                        "input_payload": {"query": "boots"},
                    }, {}, None, "",
                ),
                ConversationMessage(
                    uuid4(), uuid4(), 1, "tool_result", "internal", "tool", "", {
                        "tool_name": "products_search",
                        "provider_tool_call_id": "call-1",
                        "status": "completed",
                        "evidence": [],
                    }, {}, tool_call_id, "",
                ),
            ],
        ),
    )
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    messages = prompt.build_messages()

    assert messages[0]["tool_calls"][0]["function"]["name"] == "products_search"
