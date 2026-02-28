import streamlit as st


def debug_render_message(content, content_title: str) -> None:
    with st.chat_message("assistant", avatar=":material/edit:"):
        st.markdown(content_title)
        with st.expander("Debug"):
            st.json(content)
