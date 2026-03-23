from __future__ import annotations

from langsmith import traceable

from agents.agentstate.model import AgentState, ClassificationResults
from agents.classify.prompts.classify_prompt import build_classify_prompt
from rendering.debug import build_classify_status_message, emit_status_message

# as our toolset grows we will need a classification edge so starting now.
@traceable(name="Classifier Node")
def classify(agent_state: AgentState) -> AgentState:
    prompt = build_classify_prompt(agent_state)
    raw = agent_state.llm.invoke(prompt).content.strip()

    try:
        agent_state.classification_results = ClassificationResults.model_validate_json(raw)
    except Exception:
        agent_state.classification_results = ClassificationResults()

    emit_status_message(
        build_classify_status_message(
            agent_state.classification_results.applicable_tool_categories,
            agent_state.classification_results.can_answer_confidence,
        )
    )
    if agent_state.classification_results.can_answer_confidence >= 0.8:
        agent_state.goal_reached = True

    return agent_state
