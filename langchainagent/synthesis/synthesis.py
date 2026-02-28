from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

from langchainagent.agentstate.model import AgentState
from langchainagent.synthesis.synthesis_prompt import SOLVER_PROMPT
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1", temperature=0)


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState) -> AgentState:
    if not state.iteration_trace:
        state.result = ""
        return state

    lines = []
    for iteration in state.iteration_trace:
        for step in iteration.steps:
            evidence = iteration.results.get(step.ref, "")
            lines.append(f"Plan: {step.thought}\nEvidence {step.ref}: {evidence}\n")
    prompt = SOLVER_PROMPT.format(plan_block="\n".join(lines), task=state.task)
    answer = llm.invoke(prompt).content.strip()
    state.result = answer
    return state
