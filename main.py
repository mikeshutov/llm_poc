from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from dotenv import load_dotenv
import streamlit as st

from request_orchestrator.service import run_request_orchestrator_for_query
from common.config import CONTENT_KEY, ROLE_KEY, ROLE_USER
from conversation.models.conversation_models import ConversationMetadata
from conversation.models.replay_models import PreparedReplayConversation
from conversation.repository.repo_factory import get_conversation_repo
from conversation.replay import execute_replay, prepare_replay
from personalization.profile.repository.repo_factory import get_user_profile_repo
from rendering.feedback import FEEDBACK_TARGET_KEY, clear_feedback_state, render_feedback_dialog
from rendering.replay import clear_replay_state, pop_replay_target
from rendering.file_upload import render_file_upload
from rendering.messages.chat import append_assistant_response, render_messages
from rendering.rendering import render_message
from rendering.sources import clear_sources_panel, get_sources_panel_request, render_sources_panel
from rendering.sidebar import clear_conversation_model_config_dialog, render_sidebar, request_conversation_model_config_dialog

load_dotenv()

st.set_page_config(
    page_title="LLM Agentic Chat",
    page_icon=":robot_face:",
    layout="wide",
)

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
</style>""",
    unsafe_allow_html=True,
)

conversation_repository = get_conversation_repo()
user_profile_repository = get_user_profile_repo()
qp = st.query_params
cid = qp.get("cid")
uid = qp.get("uid")
PENDING_REPLAY_KEY = "pending_replay"
PENDING_REPLAY_PREPARE_KEY = "pending_replay_prepare"


def ensure_selected_user_id() -> str:
    profiles = user_profile_repository.list_profiles(limit=100)
    if not profiles:
        user_profile_repository.ensure_profile("anonymous", display_name="Anonymous")
        profiles = user_profile_repository.list_profiles(limit=100)

    selected_user_id = str(uid).strip() if uid else st.session_state.get("selected_user_id")
    available_user_ids = {profile.user_id for profile in profiles}
    if not selected_user_id or selected_user_id not in available_user_ids:
        selected_user_id = profiles[0].user_id

    st.session_state.selected_user_id = selected_user_id
    st.query_params["uid"] = selected_user_id
    return selected_user_id


def setup_conversation(cid_value, user_id: str):
    current = st.session_state.get("conversation_id")
    if cid_value:
        conversation = conversation_repository.get_conversation(UUID(str(cid_value)))
        if conversation is not None and conversation.user_id == user_id:
            if current != cid_value:
                clear_feedback_state()
                clear_replay_state()
                clear_sources_panel()
                clear_conversation_model_config_dialog()
            st.session_state.conversation_id = str(cid_value)
            st.query_params["cid"] = str(cid_value)
            return

    clear_feedback_state()
    clear_replay_state()
    clear_sources_panel()
    clear_conversation_model_config_dialog()
    latest = conversation_repository.get_latest_conversation(user_id)
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        st.session_state.conversation_id = str(
            conversation_repository.create_conversation(
                user_id=user_id,
                metadata=ConversationMetadata(source="streamlit"),
            ).id
        )
    st.query_params["uid"] = user_id
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
        agent_result, roundtrip = run_request_orchestrator_for_query(
            conversation_id=st.session_state.conversation_id,
            user_query=final_user_query,
            user_id=st.session_state.get("selected_user_id"),
        )

    if st.session_state.messages and st.session_state.messages[-1].get(ROLE_KEY) == ROLE_USER:
        st.session_state.messages[-1]["roundtrip_id"] = str(roundtrip.id)

    append_assistant_response(
        st.session_state.conversation_id,
        final_user_query,
        agent_result,
        roundtrip=roundtrip,
    )


selected_user_id = ensure_selected_user_id()
setup_conversation(cid, selected_user_id)

replay_target = pop_replay_target()
if replay_target:
    replay_context = prepare_replay(
        replay_target["roundtrip_id"],
        user_id=selected_user_id,
    )
    clear_feedback_state()
    clear_replay_state()
    clear_sources_panel()
    st.session_state.conversation_id = replay_context.conversation_id
    st.query_params["cid"] = replay_context.conversation_id
    st.session_state.loaded_cid = None
    st.session_state.messages = []
    st.session_state.debug_turns = []
    replay_conversation = conversation_repository.get_conversation(UUID(replay_context.conversation_id))
    if replay_conversation is not None:
        st.session_state.selected_user_id = replay_conversation.user_id
        st.query_params["uid"] = replay_conversation.user_id
    replay_title = (replay_conversation.title or "Replay").strip() if replay_conversation else "Replay"
    request_conversation_model_config_dialog(
        conversation_id=replay_context.conversation_id,
        title=replay_title,
        replay_source_roundtrip_id=replay_context.source_roundtrip_id,
        replay_context=replay_context,
    )
    st.rerun()

pending_replay_prepare = st.session_state.get(PENDING_REPLAY_PREPARE_KEY)
if (
    pending_replay_prepare
    and pending_replay_prepare.get("conversation_id") == st.session_state.conversation_id
):
    st.session_state.pop(PENDING_REPLAY_PREPARE_KEY, None)
    st.session_state[PENDING_REPLAY_KEY] = pending_replay_prepare
    st.rerun()

pending_replay = st.session_state.get(PENDING_REPLAY_KEY)
if pending_replay and pending_replay.get("conversation_id") == st.session_state.conversation_id:
    st.session_state.pop(PENDING_REPLAY_KEY, None)
    clear_conversation_model_config_dialog()
    with st.spinner("Replaying conversation..."):
        replay_context = execute_replay(
            PreparedReplayConversation.model_validate(pending_replay)
        )
        st.session_state.loaded_cid = None
        st.session_state.messages = []
        render_messages(conversation_repository, st.session_state.conversation_id, render_message)
        run_live_turn(replay_context.user_prompt)
    st.rerun()

with st.sidebar:
    render_sidebar(conversation_repository)

render_messages(conversation_repository, st.session_state.conversation_id, render_message)
if get_sources_panel_request():
    render_sources_panel()

if st.session_state.get(FEEDBACK_TARGET_KEY):
    render_feedback_dialog(conversation_repository)

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
