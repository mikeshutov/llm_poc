from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conversation.models.conversation_models import ConversationContext, ConversationMessage
from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys


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

    messages = prompt.build_messages()
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == (
        "Answer helpfully.\n\n"
        "Use only the supplied evidence.\n\n"
        "CONVERSATION SUMMARY\n"
        '{\n  "conversation_summary": "The user is shopping for boots."\n}'
    )
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "I need waterproof boots."
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "What size do you need?"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "Find suitable options."


def test_agent_prompt_sends_tool_result_as_native_tool_messages() -> None:
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
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["name"] == "search_products"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == messages[0].tool_calls[0]["id"]
    assert messages[1].content == (
        '{\n  "tool_name": "search_products",\n  "status": "completed",\n'
        '  "error_message": null,\n  "duration_ms": null,\n'
        '  "result": {\n    "raw": "do not send this"\n  },\n'
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

    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["name"] == "products_search"
    assert isinstance(messages[1], ToolMessage)
