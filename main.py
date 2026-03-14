from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agent.service import run_agent_for_query
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER
from conversation.repository.repo_factory import get_conversation_repo
from rendering.messages.chat import append_assistant_response, render_messages
from rendering.rendering import render_message, _format_timestamp
from rendering.sidebar import render_sidebar

st.set_page_config(page_title="LLM Agentic Chat", page_icon="🤖")

st.markdown(
    """<style>
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarHeader"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stButton button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    button[data-testid="stBaseButton-secondary"] > div {
        justify-content: flex-start !important;
    }
</style>""",
    unsafe_allow_html=True,
)

conversation_repository = get_conversation_repo()

qp = st.query_params
cid = qp.get("cid")


def setup_conversation(cid):
    if cid:
        st.session_state.conversation_id = cid
    else:
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id


setup_conversation(cid)

with st.sidebar:
    render_sidebar(conversation_repository)

render_messages(conversation_repository, st.session_state.conversation_id, render_message, limit=10)

userQuery = st.chat_input("What are you looking for or trying to learn about?")
if userQuery:
    now = datetime.now(timezone.utc)
    st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: userQuery, "timestamp": now})
    with st.chat_message(ROLE_USER):
        st.write(userQuery)
        ts = _format_timestamp(now)
        if ts:
            st.caption(ts)

    with st.spinner("Thinking..."):
        agent_result, roundtrip = run_agent_for_query(
            conversation_id=st.session_state.conversation_id,
            user_query=userQuery,
        )

    append_assistant_response(
        st.session_state.conversation_id,
        userQuery,
        agent_result,
        roundtrip=roundtrip,
    )
