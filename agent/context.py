from __future__ import annotations

from dataclasses import dataclass

from agent.policy import RouteDecision
from agent.state import _AgentState
from intent_layer.models.parsed_request import ParsedRequest
from llm.clients.llm_client import LlmClient


@dataclass(frozen=True)
class AgentContext:
    conversation_entries: list[dict]
    parsed_query: ParsedRequest
    conversation_id: str
    max_turns: int
    decision: RouteDecision
    llm: LlmClient
    state: _AgentState
