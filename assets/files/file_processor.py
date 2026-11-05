from __future__ import annotations

import os
from dataclasses import dataclass

import io

import pdfplumber
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import CHUNK_ENCODING, FILES_DIR, IMAGE_MIME_PREFIX
from assets.repository.asset_chunk_repository import AssetChunkRepository
from assets.repository.asset_repository import AssetRepository
from assets.files.storage import scoped_file_path
from assets.files.file_types import FileType, IMAGE_FILE_TYPES, TEXT_FILE_TYPES
from llm.clients.llm_client import LlmClient

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass
class UploadedFile:
    name: str
    type: str
    raw_bytes: bytes


def _extract_text(file: UploadedFile) -> str:
    extension = os.path.splitext(file.name)[1].lstrip(".").lower()
    if extension == FileType.PDF:
        with pdfplumber.open(io.BytesIO(file.raw_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    if extension == FileType.DOCX:
        doc = Document(io.BytesIO(file.raw_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    return file.raw_bytes.decode("utf-8", errors="replace")


def process_uploaded_file(
    file: UploadedFile,
    *,
    user_id: str | None = None,
) -> tuple[str, str, list[str]]:
    file_path = str(scoped_file_path(file.name, user_id=user_id))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.raw_bytes)

    extension = os.path.splitext(file.name)[1].lstrip(".").lower()
    image_extensions = {file_type.value for file_type in IMAGE_FILE_TYPES}
    text_extensions = {file_type.value for file_type in TEXT_FILE_TYPES}

    if file.type.startswith(IMAGE_MIME_PREFIX) or extension in image_extensions:
        caption = LlmClient().generate_caption_from_image_file(file_path)
        chunks = [caption]
    elif extension in text_extensions:
        raw_text = _extract_text(file)
        chunks = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=CHUNK_ENCODING,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).split_text(raw_text)
    else:
        chunks = [file.raw_bytes.decode("utf-8", errors="replace")]

    stored_name = os.path.basename(file_path)
    saved_file = AssetRepository().create_file(
        name=stored_name,
        storage_path=file_path,
        mime_type=file.type,
        user_id=user_id,
    )
    AssetChunkRepository().replace_content(saved_file.id, "file", chunks)

    return stored_name, str(saved_file.id), chunks
