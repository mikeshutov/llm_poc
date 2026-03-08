import streamlit as st
from conversation.repository.conversation_repository import ConversationRepository

@st.cache_resource
def get_conversation_repo() -> ConversationRepository:
    return ConversationRepository()
