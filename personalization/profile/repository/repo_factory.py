import streamlit as st

from personalization.profile.repository.user_profile_repository import UserProfileRepository


@st.cache_resource
def get_user_profile_repo() -> UserProfileRepository:
    return UserProfileRepository()
