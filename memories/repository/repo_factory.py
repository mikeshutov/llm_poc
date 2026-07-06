import streamlit as st

from memories.repository.memory_repository import MemoryRepository


@st.cache_resource
def get_memory_repo() -> MemoryRepository:
    return MemoryRepository()
