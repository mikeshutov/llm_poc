from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agent.service import run_agent_for_query
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER
from conversation.repository.repo_factory import get_conversation_repo
from rendering.file_upload import render_file_upload
from rendering.messages.chat import append_assistant_response, render_messages
from rendering.rendering import render_message
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
render_file_upload()

userQuery = st.chat_input("What are you looking for or trying to learn about?")
if userQuery:
    now = datetime.now(timezone.utc)
    uploaded_file_id = st.session_state.pop("uploaded_file_id", None)
    uploaded_file_name = st.session_state.pop("uploaded_file_name", None)
    uploaded_file_type = st.session_state.pop("uploaded_file_type", None)
    attached_file = {"name": uploaded_file_name, "type": uploaded_file_type} if uploaded_file_id else None

    msg = {ROLE_KEY: ROLE_USER, CONTENT_KEY: userQuery, "timestamp": now, "attached_file": attached_file}
    st.session_state.messages.append(msg)
    render_message(msg)

    if uploaded_file_id:
        st.session_state.file_uploader_key = st.session_state.get("file_uploader_key", 0) + 1
        userQuery = f"{userQuery}\nuploaded file name: {uploaded_file_name}, file id: {uploaded_file_id}"

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
