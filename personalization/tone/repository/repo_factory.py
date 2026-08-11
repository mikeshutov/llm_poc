import streamlit as st

from personalization.tone.repository.tone_repository import ToneRepository


@st.cache_resource
def get_tone_repo() -> ToneRepository:
    return ToneRepository()
