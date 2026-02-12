from uuid import UUID

import streamlit as st


def render_sidebar(conversation_repository) -> None:
    st.title("LLM Powered Store/Searcher")
    st.caption("Conversation")
    st.code(st.session_state.conversation_id)

    if st.button("🗑️ Delete this conversation", type="secondary"):
        conversation_repository.delete_conversation(UUID(st.session_state.conversation_id), user_id="anonymous")
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

    if st.button("➕ New chat"):
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.session_state.messages = []
        st.session_state.debug_turns = []
        st.rerun()

    st.divider()
    st.caption("Conversations")
    conversations = conversation_repository.list_conversations(user_id="anonymous", limit=50)

    # Build labels + stable mapping
    items = []
    id_by_label = {}
    for c in conversations:
        title = (c.title or "Untitled").strip()
        label = f"{title}  ·  {str(c.id)[:8]}"
        items.append(label)
        id_by_label[label] = str(c.id)

    # Pick current index
    current_id = st.session_state.conversation_id
    current_index = 0
    for i, label in enumerate(items):
        if id_by_label[label] == current_id:
            current_index = i
            break

    selected = st.radio(
        label="Conversations",
        options=items,
        index=current_index if items else 0,
        label_visibility="collapsed",
    )

    if items:
        new_id = id_by_label[selected]
        if new_id != current_id:
            st.session_state.conversation_id = new_id
            st.query_params["cid"] = new_id
            st.session_state.loaded_cid = None
            st.rerun()
