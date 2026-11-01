from __future__ import annotations

from pathlib import Path

import pytest

from capabilities.adapters.writing.templates import (
    WritingTemplate,
    load_writing_template_content,
    load_writing_templates,
    writing_templates,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _catalog(tmp_path: Path, definitions: str, **template_files: str) -> None:
    _write(tmp_path / "definitions.yaml", "templates:\n" + definitions)
    for filename, content in template_files.items():
        _write(tmp_path / filename, content)


def test_load_writing_templates_validates_yaml_definitions(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        """
  - id: essay
    name: Essay
    description: A structured essay.
    category: academic
    formats: [docx, pdf]
    tags: [essay, research]
    required_fields: [title, topic]
    template_file: essay.md
    version: 1
""",
        **{"essay.md": "{{ body }}"},
    )

    templates = load_writing_templates(tmp_path)

    assert templates == [
        WritingTemplate(
            id="essay",
            name="Essay",
            description="A structured essay.",
            category="academic",
            formats=["docx", "pdf"],
            tags=["essay", "research"],
            required_fields=["title", "topic"],
            template_file="essay.md",
            version=1,
        )
    ]


def test_load_writing_templates_rejects_duplicate_ids(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        """
  - id: duplicate
    name: Duplicate
    description: Duplicate template.
    category: test
    formats: [md]
    tags: []
    required_fields: []
    template_file: one.md
    version: 1
  - id: duplicate
    name: Duplicate Again
    description: Duplicate template.
    category: test
    formats: [md]
    tags: []
    required_fields: []
    template_file: two.md
    version: 1
""",
        **{"one.md": "one", "two.md": "two"},
    )

    with pytest.raises(ValueError, match="Duplicate writing template id"):
        load_writing_templates(tmp_path)


def test_load_writing_templates_rejects_unsupported_formats(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        """
  - id: bad
    name: Bad
    description: Bad template.
    category: test
    formats: [html]
    tags: []
    required_fields: []
    template_file: bad.md
    version: 1
""",
        **{"bad.md": "bad"},
    )

    with pytest.raises(ValueError, match="Unsupported template format"):
        load_writing_templates(tmp_path)


def test_writing_templates_filters_and_returns_metadata_only(monkeypatch, tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        """
  - id: essay
    name: Essay
    description: A structured essay.
    category: academic
    formats: [docx, pdf]
    tags: [essay, research]
    required_fields: [title, topic]
    template_file: essay.md
    version: 1
  - id: meeting-notes
    name: Meeting Notes
    description: Notes and action items.
    category: notes
    formats: [md]
    tags: [meeting]
    required_fields: [title]
    template_file: notes.md
    version: 1
""",
        **{"essay.md": "private essay body", "notes.md": "private notes body"},
    )
    monkeypatch.setattr(
        "capabilities.adapters.writing.templates.TEMPLATES_DIR",
        tmp_path,
    )

    # Query is currently ignored; format remains an active filter.
    result = writing_templates.invoke({"query": "does-not-match", "format": "PDF"})

    assert [template.id for template in result.result.templates] == ["essay"]
    assert len(result.evidence) == 1
    assert result.evidence[0].item_id == "essay"
    assert "content" not in result.evidence[0].llm_metadata
    assert "private essay body" not in str(result)


def test_writing_templates_fails_when_directory_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "capabilities.adapters.writing.templates.TEMPLATES_DIR",
        tmp_path / "missing",
    )

    with pytest.raises(FileNotFoundError, match="Writing templates directory does not exist"):
        writing_templates.invoke({})


def test_template_body_is_loaded_only_explicitly(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        """
  - id: essay
    name: Essay
    description: A structured essay.
    category: academic
    formats: [md]
    tags: [essay]
    required_fields: [title]
    template_file: essay.md
    version: 1
""",
        **{"essay.md": "private essay body"},
    )

    assert load_writing_template_content("essay", tmp_path) == "private essay body"


def test_research_paper_template_contains_required_baseline_sections() -> None:
    from capabilities.adapters.writing.templates import TEMPLATES_DIR

    definitions = {template.id: template for template in load_writing_templates(TEMPLATES_DIR)}
    research = definitions["research-paper"]
    body = load_writing_template_content("research-paper", TEMPLATES_DIR)

    assert research.required_fields == [
        "title",
        "abstract",
        "introduction",
        "methodology",
        "findings",
        "discussion",
        "limitations",
        "conclusion",
        "references",
    ]
    assert "## Abstract" in body
    assert "## 1. Introduction" in body
    assert "## 2. Methodology" in body
    assert "## 3. Findings" in body
    assert "## 4. Discussion" in body
    assert "## 5. Limitations" in body
    assert "## 6. Conclusion" in body
    assert "## References" in body
    assert "150–250-word" in body


def test_essay_template_contains_required_baseline_sections() -> None:
    from capabilities.adapters.writing.templates import TEMPLATES_DIR

    definitions = {template.id: template for template in load_writing_templates(TEMPLATES_DIR)}
    essay = definitions["essay"]
    body = load_writing_template_content("essay", TEMPLATES_DIR)

    assert essay.required_fields == [
        "title",
        "introduction",
        "body_paragraphs",
        "conclusion",
        "references",
    ]
    assert "# {{ title }}" in body
    assert "## Introduction" in body
    assert "## Body paragraphs" in body
    assert "## Conclusion" in body
    assert "## References" in body
    assert "{{ section_heading }}" not in body
