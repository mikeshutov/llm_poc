from uuid import UUID

import streamlit as st


def _delete_conversation(conversation_repository, conversation_id: str) -> None:
    conversation_repository.delete_conversation(UUID(conversation_id), user_id="anonymous")
    latest = conversation_repository.get_latest_conversation("anonymous")
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        conv = conversation_repository.create_conversation(
            user_id="anonymous",
            metadata={"source": "streamlit"},
        )
        st.session_state.conversation_id = str(conv.id)
    st.query_params["cid"] = st.session_state.conversation_id
    st.session_state.loaded_cid = None
    st.session_state.messages = []
    st.rerun()


def render_sidebar(conversation_repository) -> None:
    st.title("LLM Agentic Chat")

    if st.button("➕ New chat"):
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id
        st.session_state.loaded_cid = None
        st.session_state.messages = []
        st.session_state.debug_turns = []
        st.rerun()

    st.divider()
    # move the random styles somewhere else later
    st.markdown(
        """<style>
        button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    st.caption("Conversations")
    conversations = conversation_repository.list_conversations(user_id="anonymous", limit=50)
    current_id = st.session_state.conversation_id

    for c in conversations:
        cid = str(c.id)
        title = (c.title or "Untitled").strip()
        is_active = cid == current_id

        col_title, col_delete = st.columns([5, 1])
        with col_title:
            if is_active:
                # move these somewhere else later on if wanna keep
                st.markdown(
                    f'<div style="'
                    f'background:rgba(99,136,219,0.2);'
                    f'border-left:3px solid #6388db;'
                    f'border-radius:4px;'
                    f'padding:6px 10px;'
                    f'font-weight:600;'
                    f'white-space:nowrap;'
                    f'overflow:hidden;'
                    f'text-overflow:ellipsis;'
                    f'">{title}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(title, key=f"conv_{cid}", use_container_width=True):
                    st.session_state.conversation_id = cid
                    st.query_params["cid"] = cid
                    st.session_state.loaded_cid = None
                    st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"del_{cid}", help="Delete conversation"):
                _delete_conversation(conversation_repository, cid)
