from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb
from pydantic import BaseModel

from common.config import ROLE_ASSISTANT, ROLE_TOOL
from common.data import bounded_json_value
from common.signatures import build_signature
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.models.plan import Plan, PlanStep
from capabilities.registry.repository.models import ToolCall


def _optional_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


class MessageToolRepository:
    """Access tool activity stored as ordered conversation messages."""

    def __init__(self):
        from conversation.repository.repo_factory import get_conversation_repo
        self._repo = get_conversation_repo()

    @staticmethod
    def _sanitize_for_storage(value: Any) -> Any:
        return bounded_json_value(value, max_bytes=16_000)

    @classmethod
    def _evidence_for_storage(cls, evidence: Any) -> list[dict[str, Any]]:
        result = []
        for item in evidence or []:
            if isinstance(item, BaseModel):
                item = item.model_dump(exclude={"raw_payload"})
            elif isinstance(item, dict):
                item = dict(item)
                item.pop("raw_payload", None)
            result.append(cls._sanitize_for_storage(item))
        return result

    def append_tool_call(self, roundtrip_id: UUID, plan: Plan | None, step: PlanStep, *, input_payload=None,
                         output_payload=None, evidence=None, status=None, error_message=None, duration_ms=None) -> UUID:
        sanitized_input = self._sanitize_for_storage(step.args if input_payload is None else input_payload)
        output = self._sanitize_for_storage(output_payload) if output_payload is not None else None
        if isinstance(output, dict):
            output.pop("evidence", None)
            output.pop("raw_payload", None)
        request_hash = build_signature({"tool_name": step.tool, "input": sanitized_input})
        request_message = self._repo.append_message(
            roundtrip_id, kind="tool_call", role=ROLE_ASSISTANT, visibility="internal",
            content=f"{step.tool}: {step.hypothesis or step.plan}".strip(),
            payload={
                "tool_name": step.tool, "input_payload": sanitized_input,
                "request_hash": request_hash,
                "plan_id": self._sanitize_for_storage(plan.db_id if plan else None),
                "plan_step_id": self._sanitize_for_storage(step.db_id),
                "provider_tool_call_id": step.id,
                "step_index": step.step_index,
                "goal": step.plan, "hypothesis": step.hypothesis,
                "call_metadata": self._sanitize_for_storage(step.metadata),
            }, metadata={"tool_name": step.tool},
        )
        result_message = self._repo.append_message(
            roundtrip_id, kind="tool_result", role=ROLE_TOOL, visibility="internal",
            # The result is represented by the structured evidence payload;
            # repeating the call hypothesis in content duplicates the tool
            # call when the message stream is converted for the LLM.
            content="",
            payload={
                "tool_name": step.tool, "input_payload": sanitized_input, "output_payload": output,
                "evidence": self._evidence_for_storage(evidence),
                "error_message": error_message,
                "status": status or ("completed" if output is not None else "pending"),
                "duration_ms": duration_ms, "request_hash": request_hash,
                "plan_id": self._sanitize_for_storage(plan.db_id if plan else None),
                "plan_step_id": self._sanitize_for_storage(step.db_id),
                "provider_tool_call_id": step.id,
                "step_index": step.step_index, "goal": step.plan, "hypothesis": step.hypothesis,
                "call_metadata": self._sanitize_for_storage(step.metadata),
            }, metadata={"tool_name": step.tool}, parent_message_id=request_message.id,
        )
        return result_message.id

    def _tool_messages_for_ids(self, message_ids: list[UUID]):
        if not message_ids:
            return []
        with self._repo._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT roundtrip_id FROM conversation_message WHERE id = ANY(%s)", (message_ids,))
            roundtrip_ids = [row["roundtrip_id"] for row in cur.fetchall()]
        messages = []
        for roundtrip_id in roundtrip_ids:
            messages.extend(message for message in self._repo.list_messages(roundtrip_id)
                            if message.kind == "tool_result" and message.id in message_ids)
        by_id = {message.id: message for message in messages}
        return [by_id[message_id] for message_id in message_ids if message_id in by_id]

    @staticmethod
    def _tool_call_from_message(message) -> ToolCall:
        payload = message.payload
        return ToolCall(
            id=message.id, roundtrip_id=message.roundtrip_id, plan_id=_optional_uuid(payload.get("plan_id")),
            plan_step_id=_optional_uuid(payload.get("plan_step_id")), step_index=payload.get("step_index"),
            tool_name=payload.get("tool_name", ""), request_hash=payload.get("request_hash", ""),
            status=payload.get("status", ""), input_payload=payload.get("input_payload") or {},
            output_payload=payload.get("output_payload"), error_message=payload.get("error_message"),
            duration_ms=payload.get("duration_ms"), goal=payload.get("goal"),
            hypothesis=payload.get("hypothesis", ""), call_metadata=payload.get("call_metadata") or {},
            summary=payload.get("summary"), created_at=message.created_at,
        )

    def has_request_hash(self, roundtrip_id: UUID, request_hash: str) -> bool:
        return any(message.payload.get("request_hash") == request_hash
                   for message in self._repo.list_messages(roundtrip_id) if message.kind == "tool_result")

    def get_tool_calls(self, tool_call_ids: list[UUID]) -> list[ToolCall]:
        return [self._tool_call_from_message(message) for message in self._tool_messages_for_ids(tool_call_ids)]

    def get_tool_results(self, tool_call_ids: list[UUID]) -> list[ToolResult]:
        results: list[ToolResult] = []
        messages = self._tool_messages_for_ids(tool_call_ids)
        for message in messages:
            call = self._tool_call_from_message(message)
            payload = call.output_payload if isinstance(call.output_payload, dict) else {}
            stored_evidence = message.payload.get("evidence", [])
            evidence: list[EvidenceView] = []
            for item in stored_evidence if isinstance(stored_evidence, list) else []:
                try:
                    evidence.append(EvidenceView.model_validate(item))
                except Exception:
                    continue

            # Output payloads are persisted as JSON, so validate the complete
            # result after restoring evidence models. This keeps the repository
            # boundary typed and prevents dicts leaking into orchestration code.
            result_payload = dict(payload)
            result_payload.update({
                "tool_call_id": call.id,
                "plan_step_id": call.plan_step_id,
                "tool_name": call.tool_name,
                "hypothesis": call.hypothesis,
                "call_metadata": call.call_metadata,
                "evidence": evidence,
            })
            try:
                results.append(ToolResult.model_validate(result_payload))
            except Exception:
                results.append(ToolResult(
                    result=payload,
                    tool_call_id=call.id,
                    plan_step_id=call.plan_step_id,
                    tool_name=call.tool_name,
                    hypothesis=call.hypothesis,
                    call_metadata=call.call_metadata,
                    evidence=evidence,
                ))
        return results

    def get_tool_calls_by_roundtrips(self, roundtrip_ids: list[UUID]) -> dict[UUID, list[ToolCall]]:
        return {roundtrip_id: [self._tool_call_from_message(message)
                for message in self._repo.list_messages(roundtrip_id) if message.kind == "tool_result"]
                for roundtrip_id in roundtrip_ids}

    def copy_tool_calls(self, roundtrip_id_map: dict[UUID, UUID]) -> None:
        for source_id, target_id in roundtrip_id_map.items():
            copied_ids: dict[UUID, UUID] = {}
            for message in self._repo.list_messages(source_id):
                if message.kind in {"tool_call", "tool_result"}:
                    copied = self._repo.append_message(target_id, kind=message.kind, role=message.role,
                        visibility=message.visibility, content=message.content,
                        payload=message.payload, metadata=message.metadata,
                        parent_message_id=copied_ids.get(message.parent_message_id))
                    copied_ids[message.id] = copied.id

    def update_tool_call_summary(self, tool_call_id: UUID, summary: str) -> None:
        for message in self._tool_messages_for_ids([tool_call_id]):
            payload = dict(message.payload)
            payload["summary"] = summary
            with self._repo._conn.cursor() as cur:
                cur.execute("UPDATE conversation_message SET payload = %s WHERE id = %s",
                            (Jsonb(payload), tool_call_id))
