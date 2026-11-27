import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conversation.models.conversation_models import ConversationContext
from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys


def test_agent_prompt_builds_native_chat_messages(conversation_message_factory) -> None:
    prompt = AgentPrompt(
        instruction="Answer helpfully.",
        rules="Use only the supplied evidence.",
        conversation_context=ConversationContext(
            conversation_summary="The user is shopping for boots.",
            messages=[
                conversation_message_factory(content="I need waterproof boots."),
                conversation_message_factory(
                    sequence=1,
                    kind="assistant",
                    role="assistant",
                    content="What size do you need?",
                ),
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


def test_agent_prompt_sends_tool_result_as_native_tool_messages(conversation_message_factory) -> None:
    call_id, roundtrip_id = uuid4(), uuid4()
    prompt = AgentPrompt(
        instruction="",
        conversation_context=ConversationContext(
            messages=[
                conversation_message_factory(
                    id=call_id,
                    roundtrip_id=roundtrip_id,
                    kind="tool_call",
                    visibility="internal",
                    role="assistant",
                    payload={"tool_name": "products_search", "input_payload": {}},
                ),
                conversation_message_factory(
                    roundtrip_id=roundtrip_id,
                    sequence=1,
                    kind="tool_result",
                    visibility="internal",
                    role="tool",
                    content="ignored hypothesis",
                    payload={
                        "tool_name": "products_search",
                        "status": "completed",
                        "output_payload": {"raw": "do not send this"},
                        "evidence": [{
                            "id": "evidence-1",
                            "summary": "Waterproof boots",
                                "data": {"title": "Boots"},
                        }],
                    },
                    parent_message_id=call_id,
                ),
            ],
        ),
    )
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    messages = prompt.build_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["name"] == "products_search"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == messages[0].tool_calls[0]["id"]
    assert messages[1].tool_call_id == str(call_id)
    assert json.loads(messages[1].content)["evidence"][0]["id"] == "evidence-1"
    assert "do not send this" not in messages[1].content


def test_agent_prompt_preserves_historical_tool_names(conversation_message_factory) -> None:
    tool_call_id = uuid4()
    prompt = AgentPrompt(
        instruction="",
        conversation_context=ConversationContext(
            messages=[
                conversation_message_factory(id=tool_call_id, kind="tool_call", visibility="internal", role="assistant", payload={
                        "tool_name": "products_search",
                        "provider_tool_call_id": "call-1",
                        "input_payload": {"query": "boots"},
                    },
                ),
                conversation_message_factory(
                    sequence=1,
                    kind="tool_result",
                    visibility="internal",
                    role="tool",
                    payload={
                        "tool_name": "products_search",
                        "provider_tool_call_id": "call-1",
                        "status": "completed",
                        "evidence": [],
                    },
                    parent_message_id=tool_call_id,
                ),
            ],
        ),
    )
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    messages = prompt.build_messages()

    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["name"] == "products_search"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == str(tool_call_id)


def test_replay_uses_canonical_pairs_even_when_stored_messages_are_interleaved(conversation_message_factory):
    roundtrip_id = uuid4()
    calls = [
        conversation_message_factory(
            roundtrip_id=roundtrip_id,
            sequence=index,
            kind="tool_call",
            visibility="internal",
            role="assistant",
            payload={"tool_name": name, "input_payload": {}},
        ) for index, name in enumerate(["web_search", "math_calculate"])
    ]
    results = [
        conversation_message_factory(
            roundtrip_id=roundtrip_id,
            sequence=index + 2,
            kind="tool_result",
            visibility="internal",
            role="tool",
            payload={"tool_name": call.payload["tool_name"], "evidence": []},
            parent_message_id=call.id,
        ) for index, call in enumerate(calls)
    ]
    prompt = AgentPrompt(instruction="", conversation_context=ConversationContext(messages=[*calls, *results]))
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    messages = prompt.build_messages()

    assert [type(message) for message in messages] == [AIMessage, ToolMessage, AIMessage, ToolMessage]
    assert [messages[index].tool_call_id for index in (1, 3)] == [str(call.id) for call in calls]


def test_replay_does_not_fabricate_calls_for_unpaired_results(conversation_message_factory):
    result = conversation_message_factory(
        kind="tool_result",
        visibility="internal",
        role="tool",
        payload={"tool_name": "web_search", "provider_tool_call_id": "legacy-call"},
    )
    prompt = AgentPrompt(instruction="", conversation_context=ConversationContext(messages=[result]))
    prompt.include_section(PromptSectionKeys.CONVERSATION_CONTEXT)

    assert prompt.build_messages() == []
