import os

from dotenv import load_dotenv
from uuid import UUID
from agent.agent import run_agent
from conversation.repository.repo_factory import get_conversation_repo
from conversation.context_builder import build_roundtrip_context
from debug import debug_render_message, emit_tool_trace_debug
from rendering.rendering import render_message
from rendering.sidebar import render_sidebar
from rendering.messages import ensure_messages_loaded, render_messages, append_assistant_response
from tool_registry import register_default_tools
from common.message_constants import CONTENT_KEY, ROLE_DEBUG, ROLE_KEY, ROLE_USER


def use_langchain_enabled() -> bool:
    return os.getenv("useLangChain", "false").strip().lower() == "true"

def setup_conversation(cid):
    if cid:
        st.session_state.conversation_id = cid
    else:
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id  # persist in URL


def prepare_query_context(user_query: str, limit: int = 5):
    roundtrips_with_latest = build_roundtrip_context(
        conversation_repository,
        st.session_state.conversation_id,
        user_query,
        limit=limit,
    )
    parsed_query = parse_query(roundtrips_with_latest)
    resolve_conversation_tone_label(
        conversation_repository=conversation_repository,
        conversation_id=UUID(st.session_state.conversation_id),
        parsed_tone=parsed_query.tone,
    )

    return roundtrips_with_latest, parsed_query


load_dotenv()
register_default_tools()
import streamlit as st
from langchainagent.agent import run_langchain_agent
from intent_layer.intent_layer import parse_query
from personalization.tone import resolve_conversation_tone_label

#Page title
st.set_page_config(page_title="Product Finder", page_icon="🛒")
#get query parameters
qp = st.query_params
cid = qp.get("cid")


conversation_repository = get_conversation_repo()

setup_conversation(cid)

# sidebar to do hold conversation list and title
with st.sidebar:
    render_sidebar(conversation_repository)

ensure_messages_loaded(
    conversation_repository,
    st.session_state.conversation_id,
    limit=10,
)

# output whole chat
render_messages(st.session_state.messages, render_message)

# prompt area
# needs to be heavily refactored but it was a good way to get started on figuring this thing out
userQuery = st.chat_input("What are you looking for? e.g. red red shirts under 50")
if userQuery:
    # store the promopt
    st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: userQuery})
    with st.chat_message(ROLE_USER):
        st.write(userQuery)

    with st.spinner("Thinking..."):
        roundtrips_with_latest, parsedQuery = prepare_query_context(userQuery, limit=5)
        queryDebugTitle = "**Debug: Parsed intent and tools**"
        st.session_state.messages.append(
            {ROLE_KEY: ROLE_DEBUG, CONTENT_KEY: parsedQuery.model_dump(), "title": queryDebugTitle}
        )
        debug_render_message(parsedQuery.model_dump(), queryDebugTitle)
        if use_langchain_enabled():

            planner_result = run_langchain_agent(
                conversation_entries=roundtrips_with_latest,
                parsed_query=parsedQuery,
                conversation_id=st.session_state.conversation_id,
                on_trace=emit_tool_trace_debug,
            )
        else:
            planner_result = run_agent(
                conversation_entries=roundtrips_with_latest,
                parsed_query=parsedQuery,
                conversation_id=st.session_state.conversation_id,
                on_trace=emit_tool_trace_debug,
        )
        debug_render_message(planner_result.debug_trace, "Agent Planner Trace")
        append_assistant_response(
            st.session_state.conversation_id,
            userQuery,
            planner_result.answer,
            parsed_query=parsedQuery.model_dump(),
            tool_traces=planner_result.tool_traces,
        )
