from __future__ import annotations

import os
from dataclasses import dataclass

import io

import pdfplumber
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.file_constants import FILES_DIR, IMAGE_MIME_PREFIX
from common.model_constants import CHUNK_ENCODING
from files.repository.file_chunk_repository import FileChunkRepository
from files.repository.file_repository import FileRepository
from llm.clients.embeddings import embed_text
from llm.clients.llm_client import LlmClient

SUPPORTED_TEXT_FILE_TYPES = ["pdf", "txt", "docx"]
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

@dataclass
class UploadedFile:
    name: str
    type: str
    raw_bytes: bytes

def _extract_text(file: UploadedFile) -> str:
    if file.name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.raw_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    if file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.raw_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    return file.raw_bytes.decode("utf-8", errors="replace")

def process_uploaded_file(
    file: UploadedFile,
) -> tuple[str, str, list[str]]:
    file_path = os.path.join(FILES_DIR, file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.raw_bytes)

    if file.type.startswith(IMAGE_MIME_PREFIX):
        caption = LlmClient().generate_caption_from_image_file(file_path)
        chunks = [caption]
    else:
        raw_text = _extract_text(file)
        chunks = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=CHUNK_ENCODING,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).split_text(raw_text)

    saved_file = FileRepository().create_file(file_path, file.name, file.type)
    embedded_chunks = [(i, chunk, embed_text(chunk)) for i, chunk in enumerate(chunks)]
    FileChunkRepository().save_chunks(saved_file.id, embedded_chunks)

    return file.name, str(saved_file.id), chunks
