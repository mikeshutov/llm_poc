from __future__ import annotations

from langsmith import traceable

from agent.agentstate.model import AgentState, flatten_conversation_entries
from agent.classify.classify import classify
from agent.executor.executor import run_executor
from agent.graph_constants import CLASSIFICATION_EDGE, EXECUTE_TOOLS_EDGE, PLAN_EDGE, SYNTHESIZE_EDGE
from agent.models.agent_result import AgentResult
from agent.planner.planner import run_planner
from agent.router.router import router
from agent.synthesis.synthesis import run_synthesis
from agent.validator.validator import validator
from langgraph.graph import END, StateGraph


@traceable(name="Main Agent")
def run_agent(
    conversation_entries: list[dict],
    conversation_id: str,
    max_turns: int = 10,
) -> AgentResult:
    #Agent state and set up hybrid ReWoo Loop
    agentState = AgentState.new(
        task=flatten_conversation_entries(conversation_entries),
        max_turns=max_turns,
        conversation_entries=conversation_entries,
    )
    builder = StateGraph(AgentState)
    builder.add_node(CLASSIFICATION_EDGE, classify)
    builder.add_node(PLAN_EDGE, run_planner)
    builder.add_node(EXECUTE_TOOLS_EDGE, run_executor)
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(CLASSIFICATION_EDGE)
    
    # Check for potentially already being ready to answer during classification
    builder.add_conditional_edges(
        CLASSIFICATION_EDGE,
        router,
        {
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
            PLAN_EDGE: PLAN_EDGE,  
        },
    )

    builder.add_conditional_edges(
        PLAN_EDGE,
        validator,
        {
            EXECUTE_TOOLS_EDGE: EXECUTE_TOOLS_EDGE, 
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE},
    )

    builder.add_conditional_edges(
        EXECUTE_TOOLS_EDGE,
        router,
        {
            PLAN_EDGE: PLAN_EDGE,
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
        },
    )

    builder.add_edge(SYNTHESIZE_EDGE, END)
    agent_graph = builder.compile()

    #create a graph to see what our chain looks like
    # png = agent_graph.get_graph(xray=1).draw_mermaid_png(
    #     background_color="white"
    # )
    # with open("graph.png", "wb") as f:
    #     f.write(png)

    final_state = agent_graph.invoke(
        agentState,
        config={"configurable": {"thread_id": conversation_id}},
    )

    final = final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
    if final.result is None:
        raise ValueError("Agent finished without setting state.result")
    
    return final.result
