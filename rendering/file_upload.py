from __future__ import annotations

import streamlit as st

from files.file_processor import SUPPORTED_TEXT_FILE_TYPES, SUPPORTED_IMAGE_TYPES, UploadedFile, process_uploaded_file

# simple render/upload function just so we can easily upload more files.
def render_file_upload() -> None:
    uploaded_file = st.file_uploader("Upload a file", type=[*SUPPORTED_TEXT_FILE_TYPES, *SUPPORTED_IMAGE_TYPES], label_visibility="collapsed")
    if uploaded_file:
        file_name, chunks = process_uploaded_file(
            file=UploadedFile(name=uploaded_file.name, type=uploaded_file.type, raw_bytes=uploaded_file.getvalue()),
        )
        st.session_state.uploaded_file_name = file_name
        st.success(f"{file_name} uploaded successfully — {len(chunks)} chunks.")
