import streamlit as st
import psycopg
import os
from psycopg.rows import dict_row
from conversation.repository.conversation_repository import ConversationRepository

@st.cache_resource
def get_conversation_repo() -> ConversationRepository:
    conn = psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    )
    conn.autocommit = True
    return ConversationRepository(conn)
