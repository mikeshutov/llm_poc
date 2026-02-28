import os

import psycopg
import streamlit as st
from psycopg.rows import dict_row

from agent.repository.tool_call_repository import ToolCallRepository


@st.cache_resource
def get_tool_call_repo() -> ToolCallRepository:
    conn = psycopg.connect(
        os.environ["DATABASE_URL"],
        row_factory=dict_row,
    )
    conn.autocommit = True
    return ToolCallRepository(conn)
