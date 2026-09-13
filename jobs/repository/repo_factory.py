import streamlit as st

from jobs.repository.job_repository import JobRepository
from jobs.repository.job_result_repository import JobResultRepository


@st.cache_resource
def get_job_repo() -> JobRepository:
    return JobRepository()


@st.cache_resource
def get_job_result_repo() -> JobResultRepository:
    return JobResultRepository()
