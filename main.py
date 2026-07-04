from __future__ import annotations

from datetime import datetime, timezone

from dotenv import load_dotenv
import streamlit as st

from agent.service import run_agent_for_query
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER
from conversation.repository.repo_factory import get_conversation_repo
from conversation.replay import populate_replay_conversation, prepare_replay_conversation
from rendering.feedback import FEEDBACK_TARGET_KEY, clear_feedback_state, render_feedback_dialog
from rendering.replay import clear_replay_state, pop_replay_target
from rendering.file_upload import render_file_upload
from rendering.messages.chat import append_assistant_response, render_messages
from rendering.rendering import render_message
from rendering.sidebar import render_sidebar

load_dotenv()

st.set_page_config(page_title="LLM Agentic Chat", page_icon=":robot_face:")

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
PENDING_REPLAY_KEY = "pending_replay"


def setup_conversation(cid):
    current = st.session_state.get("conversation_id")
    if cid:
        if current != cid:
            clear_feedback_state()
            clear_replay_state()
        st.session_state.conversation_id = cid
    else:
        clear_feedback_state()
        clear_replay_state()
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id


def run_live_turn(user_query: str, attached_file: dict | None = None) -> None:
    now = datetime.now(timezone.utc)
    msg = {
        ROLE_KEY: ROLE_USER,
        CONTENT_KEY: user_query,
        "timestamp": now,
        "attached_file": attached_file,
        "roundtrip_id": None,
    }
    st.session_state.messages.append(msg)
    render_message(msg)

    final_user_query = user_query
    if attached_file and attached_file.get("id"):
        final_user_query = (
            f"{user_query}\n"
            f"uploaded file name: {attached_file['name']}, file id: {attached_file['id']}"
        )

    with st.spinner("Thinking..."):
        agent_result, roundtrip = run_agent_for_query(
            conversation_id=st.session_state.conversation_id,
            user_query=final_user_query,
        )

    if st.session_state.messages and st.session_state.messages[-1].get(ROLE_KEY) == ROLE_USER:
        st.session_state.messages[-1]["roundtrip_id"] = str(roundtrip.id)

    append_assistant_response(
        st.session_state.conversation_id,
        final_user_query,
        agent_result,
        roundtrip=roundtrip,
    )


setup_conversation(cid)

replay_target = pop_replay_target()
if replay_target:
    replay_context = prepare_replay_conversation(replay_target["roundtrip_id"])
    clear_feedback_state()
    clear_replay_state()
    st.session_state.conversation_id = replay_context["conversation_id"]
    st.query_params["cid"] = replay_context["conversation_id"]
    st.session_state.loaded_cid = None
    st.session_state.messages = []
    st.session_state.debug_turns = []
    st.session_state[PENDING_REPLAY_KEY] = replay_context
    st.rerun()

with st.sidebar:
    render_sidebar(conversation_repository)

render_messages(conversation_repository, st.session_state.conversation_id, render_message, limit=10)
if st.session_state.get(FEEDBACK_TARGET_KEY):
    render_feedback_dialog(conversation_repository)

pending_replay = st.session_state.get(PENDING_REPLAY_KEY)
if pending_replay and pending_replay.get("conversation_id") == st.session_state.conversation_id:
    with st.spinner("Replaying conversation..."):
        replay_context = populate_replay_conversation(
            pending_replay["conversation_id"],
            pending_replay["source_roundtrip_id"],
        )
        st.session_state.pop(PENDING_REPLAY_KEY, None)
        st.session_state.loaded_cid = None
        st.session_state.messages = []
        render_messages(conversation_repository, st.session_state.conversation_id, render_message, limit=10)
        run_live_turn(replay_context["user_prompt"])

render_file_upload()

userQuery = st.chat_input("What are you looking for or trying to learn about?")
if userQuery:
    uploaded_file_id = st.session_state.pop("uploaded_file_id", None)
    uploaded_file_name = st.session_state.pop("uploaded_file_name", None)
    uploaded_file_type = st.session_state.pop("uploaded_file_type", None)
    attached_file = (
        {"id": uploaded_file_id, "name": uploaded_file_name, "type": uploaded_file_type}
        if uploaded_file_id
        else None
    )

    if uploaded_file_id:
        st.session_state.file_uploader_key = st.session_state.get("file_uploader_key", 0) + 1

    run_live_turn(userQuery, attached_file)
