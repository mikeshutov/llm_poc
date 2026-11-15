from __future__ import annotations

import streamlit as st

from assets.files.file_processor import UploadedFile, process_uploaded_file
from assets.files.file_types import IMAGE_FILE_TYPES, TEXT_FILE_TYPES


# simple render/upload function just so we can easily upload more files.
def render_file_upload() -> None:
    key = f"file_uploader_{st.session_state.get('file_uploader_key', 0)}"
    supported_types = [
        *(file_type.value for file_type in TEXT_FILE_TYPES),
        *(file_type.value for file_type in IMAGE_FILE_TYPES),
    ]
    uploaded_file = st.file_uploader("Upload a file", type=supported_types, label_visibility="collapsed", key=key)
    if uploaded_file:
        if not st.session_state.get("uploaded_file_id"):
            user_id = st.session_state.get("selected_user_id")
            file_name, file_id, chunks = process_uploaded_file(
                file=UploadedFile(name=uploaded_file.name, type=uploaded_file.type, raw_bytes=uploaded_file.getvalue()),
                user_id=user_id,
            )
            st.session_state.uploaded_file_name = file_name
            st.session_state.uploaded_file_id = file_id
            st.session_state.uploaded_file_type = uploaded_file.type
            st.success(f"{file_name} uploaded successfully - {len(chunks)} chunks.")
        else:
            st.success(f"{uploaded_file.name} ready.")
