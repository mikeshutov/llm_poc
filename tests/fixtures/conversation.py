from copy import deepcopy
from uuid import uuid4

import pytest

from conversation.models.conversation_models import (
    Conversation, ConversationMessage, ConversationRoundtrip, ConversationSummary,
)
from conversation.repository.conversation_repository import ConversationRepository


class RecordingConversationRepository:
    def __init__(self, *, llm_calls=(), conversation_events=(), messages=()) -> None:
        self.llm_calls = list(llm_calls)
        self.conversation_events = list(conversation_events)
        self.messages = list(messages)

    def record_llm_usage(self, **kwargs):
        self.llm_calls.append(kwargs)
        return kwargs

    def append_node_log(self, scope_type, scope_id, **kwargs):
        payload = kwargs.get("payload") or {}
        node_name = kwargs.get("node_name") or payload.get("agent_name") or payload.get("source", "")
        self.conversation_events.append({
            "scope_type": scope_type,
            "scope_id": scope_id,
            "roundtrip_id": scope_id if scope_type == "roundtrip" else None,
            "event_type": kwargs.get("event_type"),
            "source": node_name,
            "agent_name": node_name,
            "node_name": node_name,
            "payload": payload,
        })
        return type("RecordedLog", (), {"id": uuid4()})()

    def append_message(self, roundtrip_id, **kwargs):
        message = {"roundtrip_id": roundtrip_id, **kwargs}
        self.messages.append(message)
        payload = kwargs.get("payload") or {}
        if payload.get("kind"):
            self.conversation_events.append({
                "roundtrip_id": roundtrip_id,
                "event_type": payload["kind"],
                "source": payload.get("source", ""),
                "agent_name": payload.get("agent_name", ""),
                "node_name": payload.get("node_name", ""),
                "iteration": payload.get("iteration"),
                "payload": payload,
            })
        return type("RecordedMessage", (), {"id": uuid4()})()

    def create_roundtrip_prompt(self, *args, **kwargs):
        return None


@pytest.fixture
def recording_conversation_repository_factory():
    return RecordingConversationRepository


@pytest.fixture
def conversation_message_factory():
    def create(**overrides):
        return ConversationMessage(**deepcopy({
            "id": uuid4(), "roundtrip_id": uuid4(), "sequence": 0,
            "kind": "user", "visibility": "visible", "role": "user",
            "content": "", "payload": {}, "metadata": {},
            "parent_message_id": None, "created_at": "", **overrides,
        }))

    return create


@pytest.fixture
def conversation_tool_exchange_factory(conversation_message_factory):
    def create(*, tool="web_search", roundtrip_id=None, sequence=0, args=None, compacted=False,
               is_subagent=False, subagent_name=None, call_overrides=None, result_overrides=None):
        common = {
            "roundtrip_id": uuid4() if roundtrip_id is None else roundtrip_id,
            "visibility": "internal", "compacted": compacted,
            "is_subagent": is_subagent, "subagent_name": subagent_name,
        }
        call = conversation_message_factory(**{
            **common, "kind": "tool_call", "role": "assistant", "sequence": sequence,
            "payload": {"tool_name": tool, "input_payload": deepcopy(args or {})},
            **deepcopy(call_overrides or {}),
        })
        result = conversation_message_factory(**{
            **common, "roundtrip_id": call.roundtrip_id,
            "kind": "tool_result", "role": "tool", "sequence": call.sequence + 1,
            "payload": {"tool_name": tool}, "parent_message_id": call.id,
            **deepcopy(result_overrides or {}),
        })
        return call, result

    return create


@pytest.fixture
def conversation_roundtrip_factory():
    def create(**overrides):
        return ConversationRoundtrip(**deepcopy({
            "id": uuid4(), "conversation_id": uuid4(), "message_index": 1,
            "roundtrip_summary": "", "roundtrip_summary_embedding": None,
            "created_at": "2026-01-01", "metadata": {}, "status": "completed", **overrides,
        }))

    return create


@pytest.fixture
def conversation_repository_factory(database_connection_factory):
    """Build a real repository around a mocked DB connection, without opening one."""
    def create(*, rows=(), row=None):
        connection, cursor = database_connection_factory(rows=rows, row=row)
        repository = object.__new__(ConversationRepository)
        repository._conn = connection
        return repository, cursor

    return create


class ContextConversationRepository:
    def __init__(self, *, roundtrips, messages):
        self.roundtrips = list(roundtrips)
        self.roundtrip = self.roundtrips[0]
        self.messages = list(messages)
        self.message_queries = []

    def get_conversation(self, conversation_id):
        return Conversation(conversation_id, "user-1", "Test", "", {}, {}, "", None)

    def get_latest_summary(self, conversation_id):
        return ConversationSummary(uuid4(), conversation_id, "", 0, "")

    def list_roundtrips(self, *args, **kwargs):
        return list(self.roundtrips)

    def list_messages(self, roundtrip_id, *, include_internal=True, exclude_subagents=False, exclude_compacted=False):
        self.message_queries.append((roundtrip_id, exclude_subagents, exclude_compacted))
        return [
            message for message in self.messages
            if message.roundtrip_id == roundtrip_id
            and (include_internal or message.visibility == "visible")
            and (not exclude_subagents or not message.is_subagent)
            and (not exclude_compacted or not message.compacted)
        ]


@pytest.fixture
def context_repository_factory(conversation_roundtrip_factory, conversation_message_factory):
    def create(*, roundtrips=None, messages=None):
        roundtrips = [conversation_roundtrip_factory(roundtrip_summary="summary")] if roundtrips is None else roundtrips
        roundtrip = roundtrips[0]
        if messages is None:
            messages = [
                conversation_message_factory(roundtrip_id=roundtrip.id, content="Find boots."),
                conversation_message_factory(
                    roundtrip_id=roundtrip.id, sequence=1, kind="assistant", role="assistant",
                    content="Here are boots.", payload={"next_question": "What size?"},
                    metadata={"stage": "synthesis"},
                ),
            ]
        return ContextConversationRepository(roundtrips=roundtrips, messages=messages)

    return create


class RequestConversationRepository:
    def __init__(self):
        self.pending_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.conversation_id = uuid4()
        self.roundtrip_id = uuid4()

    def get_conversation(self, conversation_id):
        assert conversation_id == self.conversation_id
        return Conversation(
            id=self.conversation_id,
            user_id='anonymous',
            title='Test Conversation',
            created_at='2026-08-08T00:00:00Z',
            metadata={},
            tone_state={},
            summary='',
            summary_embedding=None,
        )

    def create_pending_roundtrip(self, conversation_id, user_prompt, model=None, roundtrip_summary=None, roundtrip_summary_embedding=None, metadata=None):
        self.pending_calls.append(
            {
                'conversation_id': conversation_id,
                'user_prompt': user_prompt,
                'model': model,
                'metadata': metadata,
            }
        )
        return ConversationRoundtrip(
            id=self.roundtrip_id,
            conversation_id=conversation_id,
            message_index=0,
            roundtrip_summary=None,
            roundtrip_summary_embedding=None,
            created_at='2026-08-08T00:00:00Z',
            metadata=metadata or {},
            status='pending',
        )

    def update_roundtrip(self, roundtrip_id, response, payload, roundtrip_summary=None, roundtrip_summary_embedding=None):
        self.update_calls.append(
            {
                'roundtrip_id': roundtrip_id,
                'response': response,
                'payload': payload,
                'roundtrip_summary': roundtrip_summary,
            }
        )
        return ConversationRoundtrip(
            id=roundtrip_id,
            conversation_id=self.conversation_id,
            message_index=0,
            roundtrip_summary=roundtrip_summary,
            roundtrip_summary_embedding=roundtrip_summary_embedding,
            created_at='2026-08-08T00:00:00Z',
            metadata=self.pending_calls[0]['metadata'],
            status='completed',
        )


class ConversationModelConfigRepositoryStub:
    def __init__(self, config, conversation_id):
        self.config = config
        self.conversation_id = conversation_id

    def resolve(self, conversation_id):
        assert conversation_id == self.conversation_id
        return self.config

@pytest.fixture
def request_conversation_repository_factory():
    return RequestConversationRepository


@pytest.fixture
def conversation_model_config_repository_factory():
    return ConversationModelConfigRepositoryStub
