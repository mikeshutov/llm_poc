from dataclasses import dataclass
from typing import Any

from agent.state import _AgentState


@dataclass(frozen=True)
class GovernorAction:
    action: str  # none | continue | stop
    stop_reason: str | None = None
    mark_stop_reason: str | None = None


class LoopGovernor:
    def after_turn(
        self,
        *,
        state: _AgentState,
        turn_index: int,
        allow_web_fallback: bool,
        step_tool_history: list[dict[str, Any]],
        previous_query_text: str,
        previous_filters_dump: dict[str, Any] | None,
        skipped_status: str,
    ) -> GovernorAction:
        return GovernorAction(action="none")
