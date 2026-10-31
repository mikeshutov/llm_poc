from __future__ import annotations

from agent_runtime.models.agent_state import AgentState
from agent_runtime.strategies.react.prompt import build_decision_prompt


def build_reasoner_prompt(state: AgentState):
    return build_decision_prompt(
        state,
        instruction=(
            "You are the main request reasoner. Use available tools or agents to gather the "
            "information needed to answer the user's task. If more work is needed, emit tool "
            "calls. If the task is complete or no useful action remains, emit no tool calls and "
            "return the final answer as Markdown. Use headings, lists, tables, and links when "
            "they improve readability. Do not wrap the Markdown in JSON or code fences."
        ),
        rules=(
            "Do not invent tools or agents. Use discover_capabilities when the needed capability "
            "is not currently listed. A tool call means continue; no tool calls means produce the "
            "final structured answer. Tool calls may be planned across multiple reasoner turns; "
            "use evidence from completed calls when deciding on follow-up calls."
        ),
        include_evidence=False,
    )


__all__ = ["build_reasoner_prompt"]
