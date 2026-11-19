from __future__ import annotations

import json

from langchain_core.tools import BaseTool

from agent_runtime.models.agent_prompt import AgentPrompt
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentName
from agent_runtime.models.decision import DecisionResponse
from agent_runtime.planning.prompts.planner_rules import REACT_DECISION_RULES
from agent_runtime.strategies.react.decision import build_decision_prompt


REACT_DECISION_SCHEMA = json.dumps(
    DecisionResponse.model_json_schema(),
    indent=2,
)


REACT_DECISION_INSTRUCTION = (
    "You are an action-oriented decision agent. Reason over the user's goal and the "
    "observations already collected. Choose the next useful tool actions, or return "
    "no actions when the goal is satisfied or no useful action remains. "
    "Use only facts, source details, and URLs supported by the observed evidence; if "
    "evidence is insufficient, choose a research action instead of guessing. Native "
    "tool calls are the authoritative actions."
)

REACT_DECISION_CONTRACT_INSTRUCTION = (
    "Always return the structured decision response described in the output contract. "
    "Use status to report whether the task is continuing, satisfied, or blocked, and use "
    "message to briefly explain the decision."
)


def build_subagent_decision_prompt(
    state: AgentState,
) -> tuple[AgentPrompt, list[BaseTool]]:
    instruction = REACT_DECISION_INSTRUCTION
    schema = REACT_DECISION_SCHEMA
    if state.agent_profile.name != AgentName.DELEGATE:
        instruction = f"{instruction} {REACT_DECISION_CONTRACT_INSTRUCTION}"
    else:
        # Delegate completion is represented by the runtime's tool result and
        # collected evidence; it does not need a second JSON response envelope.
        schema = ""
    if state.agent_profile.planner_instruction:
        instruction = f"{instruction}\n\nAgent-specific instruction:\n{state.agent_profile.planner_instruction}"
    rules = [*REACT_DECISION_RULES]
    return build_decision_prompt(
        state,
        instruction=instruction,
        rules="\n".join(f"- {rule}" for rule in rules),
        schema=schema,
    )


__all__ = [
    "build_subagent_decision_prompt",
]
