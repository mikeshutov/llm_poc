from __future__ import annotations
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable
import re

from langchainagent.agentstate.model import AgentState, IterationState, PlanStep
from langchainagent.planner import planner_prompt
from langchainagent.tools import LANGCHAIN_TOOLS
load_dotenv()

# for now this is here
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

tool_list = "\n".join(
    f'- {tool.name}: {getattr(tool, "description", "")}'.strip()
    for tool in LANGCHAIN_TOOLS
)
## for now we just regex the plan out
plan_regex = r"Plan:\s*(.*?)\s*#E(\d+)\s*=\s*(\w+)\s*(\{.*?\})\s*(?=Plan:|$)"

def parse_steps(raw_plan: str) -> list[PlanStep]:
    matches = re.findall(plan_regex, raw_plan, flags=re.S)
    return [PlanStep(thought=m[0].strip(), ref=m[1], tool=m[2], args=m[3].strip()) for m in matches]

@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()
    raw_plan = llm.invoke(
        planner_prompt.PLANNER_PROMPT.format(tool_list=tool_list, task=agent_state.task)
    ).content
    it_state.plan_string = raw_plan
    it_state.steps = parse_steps(raw_plan)
    agent_state.add_iteration(it_state)

    print(raw_plan)
    if not it_state.steps:
        # maybe add: agent_state.error = f"Planner produced no steps. Raw plan: {raw_plan}"
        agent_state.goal_reached = True
    return agent_state
