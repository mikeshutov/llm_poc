from __future__ import annotations

import importlib
from uuid import uuid4

from agent_runtime.context import bind_agent_context
from capabilities.adapters.memories.get_memory_detail import get_memory_detail


def test_working_memory_tool_automatically_condenses_current_roundtrip() -> None:
    module = importlib.import_module("capabilities.adapters.workspace.working_memory")
    tool_description = module.update_memory_state.description

    assert "task state and working memory" in tool_description
    assert "objective, progress, findings" in tool_description
    assert "remaining work" in tool_description
    assert "automatically condenses previously viewed compactable tool exchanges" in tool_description
    assert "current roundtrip" in tool_description
    assert "source_tool_call_ids" not in module.UpdateMemoryStateArgs.model_fields
    assert "discard_tool_call_ids" not in module.UpdateMemoryStateArgs.model_fields
    assert "task_completed" in module.UpdateMemoryStateArgs.model_fields


def test_workspace_write_and_edit_only_update_files(runtime_workspace) -> None:
    module = importlib.import_module("capabilities.adapters.workspace.files")

    write_result = module.workspace_file_write.invoke({
        "file_name": "notes.md",
        "content": "before",
    })
    assert "source_tool_call_ids" not in write_result.result
    assert "condensed_tool_call_ids" not in write_result.result

    edit_result = module.workspace_file_edit.invoke({
        "file_name": "notes.md",
        "find": "before",
        "replace": "after",
    })
    assert "source_tool_call_ids" not in edit_result.result
    assert "condensed_tool_call_ids" not in edit_result.result
    assert (runtime_workspace / "notes.md").read_text(encoding="utf-8") == "after"


def test_get_memory_detail_returns_typed_result(
    conversation_roundtrip_factory,
    conversation_message_factory,
    agent_state_factory,
) -> None:
    class FakeConversationRepo:
        def __init__(self) -> None:
            self.roundtrip = conversation_roundtrip_factory(
                id=uuid4(),
                conversation_id=uuid4(),
                message_index=7,
                roundtrip_summary="The conversation decided on a 30-day return policy.",
                roundtrip_summary_embedding=None,
                created_at="2026-08-01T12:00:00+00:00",
                metadata={"model": "gpt-5.6-terra"},
                status="completed",
            )
            self.messages = [
                conversation_message_factory(
                    roundtrip_id=self.roundtrip.id,
                    content="What did we decide about returns?",
                ),
                conversation_message_factory(
                    roundtrip_id=self.roundtrip.id,
                    sequence=1,
                    kind="assistant",
                    role="assistant",
                    content="We said we would offer 30-day returns.",
                    metadata={"stage": "synthesis", "model": "gpt-5.6-terra"},
                ),
            ]

        def get_roundtrip_for_user(self, roundtrip_id, user_id: str | None):
            assert str(roundtrip_id) == str(self.roundtrip.id)
            assert user_id == "user-123"
            return self.roundtrip

        def list_messages(self, roundtrip_id, include_internal=True):
            return self.messages

    module = importlib.import_module(
        "capabilities.adapters.memories.get_memory_detail"
    )
    fake_repo = FakeConversationRepo()
    original_repo_getter = module.get_conversation_repo
    module.get_conversation_repo = lambda: fake_repo
    try:
        with bind_agent_context(agent_state_factory(user_id="user-123")):
            result = get_memory_detail.invoke({"roundtrip_id": str(fake_repo.roundtrip.id)})
    finally:
        module.get_conversation_repo = original_repo_getter

    assert result.result.memory_type == "roundtrip"
    assert result.result.roundtrip_id == str(fake_repo.roundtrip.id)
    assert result.result.summary == "The conversation decided on a 30-day return policy."
    assert result.evidence[0].data["item_id"] == str(fake_repo.roundtrip.id)
    assert result.evidence[0].data["title"] == "Memory detail for message 7"
    assert result.evidence[0].summary == "The conversation decided on a 30-day return policy."
    assert result.evidence[0].data["item_id"] == str(fake_repo.roundtrip.id)


def test_get_memory_detail_rejects_invalid_roundtrip_id() -> None:
    result = get_memory_detail.invoke({"roundtrip_id": "not-a-uuid"})

    assert result.result.error == (
        "Invalid roundtrip_id 'not-a-uuid'. Call memory_search first and use the exact "
        "roundtrip UUID returned in its evidence."
    )
    assert result.evidence == []
