from dotenv import load_dotenv
from conversation.repository.repo_factory import get_conversation_repo
from rendering.messages.chat import (
    append_assistant_response,
    render_messages,
)
from agent.service import run_agent_for_query
from rendering.rendering import render_message
from rendering.sidebar import render_sidebar
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_USER

def setup_conversation(cid):
    if cid:
        st.session_state.conversation_id = cid
    else:
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id  # persist in URL
load_dotenv()
import streamlit as st

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

# output whole chat
render_messages(conversation_repository, st.session_state.conversation_id, render_message, limit=10)

# prompt area
# needs to be heavily refactored but it was a good way to get started on figuring this thing out
userQuery = st.chat_input("What are you looking for or trying to learn about?")
if userQuery:
    # store the promopt
    st.session_state.messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: userQuery})
    with st.chat_message(ROLE_USER):
        st.write(userQuery)

    with st.spinner("Thinking..."):
        agent_result = run_agent_for_query(
            conversation_id=st.session_state.conversation_id,
            user_query=userQuery,
        )
        append_assistant_response(
            st.session_state.conversation_id,
            userQuery,
            agent_result,
        )
