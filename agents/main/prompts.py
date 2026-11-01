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
            "they improve readability. Ground factual claims, citations, URLs, and generated "
            "document content in evidence produced by the current roundtrip. If the evidence "
            "does not support a claim, gather more evidence or state that it is unavailable; "
            "do not fill the gap from general knowledge. Do not wrap the Markdown in JSON or "
            "code fences."
        ),
        rules=(
            "Do not invent tools or agents. Use discover_capabilities when the needed capability "
            "is not currently listed. A tool call means continue; no tool calls means produce the "
            "final structured answer. Tool calls may be planned across multiple reasoner turns; "
            "use evidence from completed calls when deciding on follow-up calls. Never repeat a "
            "request that was completed or rejected as a duplicate; if a request failed, only "
            "retry it with a materially corrected task or arguments. Do not invent citations, "
            "source titles, or URLs that are absent from the produced evidence."
            " For generated files, do not invent sandbox or filesystem links; the UI will "
            "render the canonical link from file evidence."
        ),
        include_evidence=False,
    )


__all__ = ["build_reasoner_prompt"]
