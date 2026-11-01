from __future__ import annotations

from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    AgentProfile,
    RESEARCH_AGENT_NAME,
)
from agents.research.prompts import build_research_rules
from llm.conversation_model_config import MAIN_AGENT_MODEL_SCOPE


def build_research_profile() -> AgentProfile:
    return AgentProfile(
        name=RESEARCH_AGENT_NAME,
        scope=MAIN_AGENT_MODEL_SCOPE,
        version=2,
        description=(
            "Research a topic by decomposing it into questions, selecting appropriate "
            "read-only sources, gathering a collection of evidence records, and refining "
            "the research plan as needed. The returned evidence can be looked up by evidence ID "
            "by a downstream agent such as the document-writing agent."
        ),
        allowed_categories={"web_search", "knowledge", "books", "language"},
        execution_strategy=AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        delegatable=True,
        max_turns=8,
        planner_instruction=(
            "Plan and execute a rigorous research workflow for the provided topic. "
            "Decompose the request, gather diverse relevant evidence, and replan when "
            "the evidence leaves an important gap. Return the collected evidence for "
            "downstream agents to look up and use. Do not answer the user directly."
        ),
        planner_rules=build_research_rules(),
    )


RESEARCH_PROFILE = build_research_profile()
