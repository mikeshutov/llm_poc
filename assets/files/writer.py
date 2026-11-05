from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Collection

from docx import Document

from assets.files.file_types import (
    DOCUMENT_FILE_TYPES,
    DOCUMENT_FILE_TYPES_DESCRIPTION,
    FileType,
)


@dataclass(frozen=True)
class PdfStyle:
    margin: int = 54
    body_font: str = "Helvetica"
    body_font_size: int = 12
    title_font: str = "Helvetica-Bold"
    title_font_size: int = 16


PDF_STYLE = PdfStyle()


def normalize_file_format(
    file_name: str,
    file_format: str | None = None,
    *,
    supported_types: Collection[FileType] = DOCUMENT_FILE_TYPES,
) -> tuple[str, FileType]:
    has_explicit_format = bool(file_format and file_format.strip())
    requested = (file_format or Path(file_name).suffix.lstrip(".") or FileType.TXT).strip().lower()
    requested = requested.lstrip(".")
    aliases = {
        "text": FileType.TXT,
        "plain text": FileType.TXT,
        "text/plain": FileType.TXT,
        "markdown": FileType.MARKDOWN,
        "text/markdown": FileType.MARKDOWN,
        "word": FileType.DOCX,
        "doc": FileType.DOCX,
        "docx": FileType.DOCX,
        "wordprocessingml": FileType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
        "pdf": FileType.PDF,
        "application/pdf": FileType.PDF,
        "svg": FileType.SVG,
        "image/svg+xml": FileType.SVG,
    }
    normalized = aliases.get(requested)
    if normalized is None:
        if "word" in requested or "wordprocessingml" in requested or "openxmlformats" in requested:
            normalized = FileType.DOCX
        elif "pdf" in requested:
            normalized = FileType.PDF
        elif "markdown" in requested:
            normalized = FileType.MARKDOWN
        elif "text" in requested:
            normalized = FileType.TXT
        else:
            try:
                normalized = FileType(requested)
            except ValueError:
                normalized = None
    if normalized is None or normalized not in supported_types:
        supported_description = (
            DOCUMENT_FILE_TYPES_DESCRIPTION
            if supported_types == DOCUMENT_FILE_TYPES
            else ", ".join(item.value for item in sorted(supported_types, key=lambda item: item.value))
        )
        raise ValueError(f"Unsupported file format. Supported formats are {supported_description}.")

    path = Path(file_name)
    if not path.suffix:
        file_name = f"{file_name}.{normalized.value}"
    elif has_explicit_format and path.suffix.lower().lstrip(".") != normalized.value:
        file_name = str(path.with_suffix(f".{normalized.value}"))
    elif not has_explicit_format and path.suffix.lower() == ".doc":
        file_name = str(path.with_suffix(f".{FileType.DOCX.value}"))
    return file_name, normalized


def _document_paragraphs(content: str) -> list[str]:
    normalized = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized:
        return [""]
    return [
        " ".join(line.strip() for line in block.splitlines()).strip()
        for block in re.split(r"\n\s*\n", normalized)
        if block.strip()
    ]


def write_file(path: Path, content: str, file_format: FileType, title: str | None = None) -> None:
    if file_format not in {FileType.DOCX, FileType.PDF}:
        path.write_text(content, encoding="utf-8")
        return
    if file_format == FileType.PDF:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import simpleSplit
            from reportlab.pdfgen.canvas import Canvas
        except ImportError as exc:
            raise ValueError("PDF creation requires the reportlab package.") from exc
        canvas = Canvas(str(path), pagesize=letter)
        width, height = letter
        style = PDF_STYLE
        text = canvas.beginText(style.margin, height - style.margin)
        text.setFont(style.body_font, style.body_font_size)
        if title and title.strip():
            text.setFont(style.title_font, style.title_font_size)
            text.textLine(title.strip())
            text.textLine("")
            text.setFont(style.body_font, style.body_font_size)
        paragraphs = _document_paragraphs(content)
        for paragraph_index, paragraph in enumerate(paragraphs):
            lines = simpleSplit(
                paragraph,
                style.body_font,
                style.body_font_size,
                width - (2 * style.margin),
            ) or [""]
            for line in lines:
                if text.getY() < style.margin:
                    canvas.drawText(text)
                    canvas.showPage()
                    text = canvas.beginText(style.margin, height - style.margin)
                    text.setFont(style.body_font, style.body_font_size)
                text.textLine(line)
            if paragraph_index < len(paragraphs) - 1:
                if text.getY() < style.margin:
                    canvas.drawText(text)
                    canvas.showPage()
                    text = canvas.beginText(style.margin, height - style.margin)
                    text.setFont(style.body_font, style.body_font_size)
                text.textLine("")
        canvas.drawText(text)
        canvas.save()
        return
    document = Document()
    if title and title.strip():
        document.add_heading(title.strip(), level=1)
    for paragraph in _document_paragraphs(content):
        document.add_paragraph(paragraph)
    document.save(path)


__all__ = [
    "PDF_STYLE",
    "PdfStyle",
    "normalize_file_format",
    "write_file",
]
