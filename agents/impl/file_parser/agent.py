from __future__ import annotations

from langgraph.graph import END, StateGraph
from langsmith import traceable

from agents.impl.file_parser.parse import parse_file
from agents.impl.file_parser.state import FileParserState
from agents.impl.file_parser.validate import validate_parse, validation_router
from agents.graph_constants import PARSE_EDGE, VALIDATE_EDGE

@traceable(name="File Parser Agent")
def run_file_parser_agent(raw_content: str, file_name: str) -> dict:
    state = FileParserState.new(raw_content=raw_content, file_name=file_name)

    builder = StateGraph(FileParserState)
    builder.add_node(PARSE_EDGE, parse_file)
    builder.add_node(VALIDATE_EDGE, validate_parse)
    builder.set_entry_point(PARSE_EDGE)
    builder.add_edge(PARSE_EDGE, VALIDATE_EDGE)
    builder.add_conditional_edges(VALIDATE_EDGE, validation_router, {
        PARSE_EDGE: PARSE_EDGE,
        VALIDATE_EDGE: END,
    })

    graph = builder.compile()
    result = graph.invoke(state)

    final = result if isinstance(result, FileParserState) else FileParserState(**result)
    return final.extracted
