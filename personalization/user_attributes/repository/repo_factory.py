import streamlit as st

from personalization.user_attributes.repository.user_attribute_repository import UserAttributeRepository


@st.cache_resource
def get_user_attribute_repo() -> UserAttributeRepository:
    return UserAttributeRepository()
