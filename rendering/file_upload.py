from __future__ import annotations

import streamlit as st

from files.file_processor import SUPPORTED_TEXT_FILE_TYPES, SUPPORTED_IMAGE_TYPES, UploadedFile, process_uploaded_file

# simple render/upload function just so we can easily upload more files.
def render_file_upload() -> None:
    key = f"file_uploader_{st.session_state.get('file_uploader_key', 0)}"
    uploaded_file = st.file_uploader("Upload a file", type=[*SUPPORTED_TEXT_FILE_TYPES, *SUPPORTED_IMAGE_TYPES], label_visibility="collapsed", key=key)
    if uploaded_file:
        if not st.session_state.get("uploaded_file_id"):
            file_name, file_id, chunks = process_uploaded_file(
                file=UploadedFile(name=uploaded_file.name, type=uploaded_file.type, raw_bytes=uploaded_file.getvalue()),
            )
            st.session_state.uploaded_file_name = file_name
            st.session_state.uploaded_file_id = file_id
            st.session_state.uploaded_file_type = uploaded_file.type
            st.success(f"{file_name} uploaded successfully — {len(chunks)} chunks.")
        else:
            st.success(f"{uploaded_file.name} ready.")
