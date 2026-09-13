import streamlit as st

from artifacts.repository.artifact_repository import ArtifactRepository


@st.cache_resource
def get_artifact_repo() -> ArtifactRepository:
    return ArtifactRepository()
