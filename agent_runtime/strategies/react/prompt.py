from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_runtime.models.agent_prompt import AgentPrompt
from agent_runtime.models.agent_state import AgentState
from agent_runtime.planning.prompts.planner_rules import REACT_DECISION_RULES
from agent_runtime.strategies.react.decision import build_decision_prompt


REACT_DECISION_INSTRUCTION = (
    "You are an action-oriented decision agent. Reason over the user's goal and the "
    "observations already collected. Choose the next useful tool actions, or return "
    "no actions when the goal is satisfied or no useful action remains. "
    "When returning no tool calls, return a JSON object with status SATISFIED or BLOCKED. "
    "A response with tool calls means CONTINUE."
)


def build_react_decision_prompt(
    state: AgentState,
) -> tuple[AgentPrompt, list[BaseTool]]:
    return build_decision_prompt(
        state,
        instruction=REACT_DECISION_INSTRUCTION,
        rules="\n".join(f"- {rule}" for rule in REACT_DECISION_RULES),
    )
