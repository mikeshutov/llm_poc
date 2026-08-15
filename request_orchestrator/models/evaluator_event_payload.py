from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm.usage import LlmCallRecord, serialize_llm_call_record


class EvaluatorEventData(BaseModel):
    status: str
    relevant_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    refined_goal: str = ""
    parse_error: str | None = None
    llm_usage: dict[str, Any] | None = None


class EvaluatorEventPayload(BaseModel):
    agent_name: str
    kind: str
    status: str
    data: EvaluatorEventData

    @classmethod
    def from_evaluation(
        cls,
        *,
        agent_name: str,
        kind: str,
        status: str,
        relevant_evidence: list[str],
        missing_information: list[str],
        refined_goal: str,
        llm_call: LlmCallRecord | None,
    ) -> "EvaluatorEventPayload":
        return cls(
            agent_name=agent_name,
            kind=kind,
            status=status,
            data=EvaluatorEventData(
                status=status,
                relevant_evidence=list(relevant_evidence),
                missing_information=list(missing_information),
                refined_goal=refined_goal,
                llm_usage=None if llm_call is None else serialize_llm_call_record(llm_call),
            ),
        )

    @classmethod
    def from_parse_error(
        cls,
        *,
        agent_name: str,
        kind: str,
        status: str,
        parse_error: str,
        llm_call: LlmCallRecord | None,
    ) -> "EvaluatorEventPayload":
        return cls(
            agent_name=agent_name,
            kind=kind,
            status=status,
            data=EvaluatorEventData(
                status=status,
                relevant_evidence=[],
                missing_information=[],
                refined_goal="",
                parse_error=parse_error,
                llm_usage=None if llm_call is None else serialize_llm_call_record(llm_call),
            ),
        )
