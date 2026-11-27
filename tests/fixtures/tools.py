from uuid import uuid4

import pytest

from capabilities.registry import global_registry
from common.signatures import build_signature


@pytest.fixture(autouse=True)
def registered_tool_registry(monkeypatch):
    """Keep the registry object stable while isolating each test's registrations."""
    registry = global_registry.GLOBAL_TOOL_REGISTRY
    monkeypatch.setattr(registry, "_tools", {})
    monkeypatch.setattr(registry, "_rate_limit_states", global_registry.ToolRegistry()._rate_limit_states)
    monkeypatch.setattr(global_registry, "_REGISTERED_DEFAULTS", False)
    global_registry.register_default_tools()
    return registry


class RecordingToolRepository:
    def __init__(self, *, records=None):
        self.records = dict(records or {})

    def count_request_hash(self, roundtrip_id, signature):
        return sum(
            record["roundtrip_id"] == roundtrip_id
            and build_signature({"tool_name": record["step"].tool, "input": record["input_payload"]}) == signature
            for record in self.records.values()
        )

    def append_tool_call(self, roundtrip_id, plan, step, **kwargs):
        call_id = uuid4()
        self.records[call_id] = {"roundtrip_id": roundtrip_id, "plan": plan, "step": step, **kwargs}
        return call_id

    def get_tool_results(self, ids):
        return [self.records[call_id]["output_payload"] for call_id in ids]


@pytest.fixture
def tool_repository_factory(monkeypatch):
    def create(*, records=None):
        repository = RecordingToolRepository(records=records)
        monkeypatch.setattr("agent_runtime.execution.tool_executor.MessageToolRepository", lambda: repository)
        monkeypatch.setattr("agent_runtime.models.agent_state.message_repository.MessageToolRepository", lambda: repository)
        return repository

    return create


@pytest.fixture
def recording_tool_repository(tool_repository_factory):
    return tool_repository_factory()
