from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from capabilities.registry.constants import (
    TOOL_NAME_WRITING_TEMPLATES,
    TOOL_RESULT_TYPE_WRITING_TEMPLATES,
)
from retrieval.evidence import EvidenceView, ToolResult


# Keep definitions beside the writing_templates capability so the tool and its
# catalog are deployed together.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_DEFINITIONS_FILE = "definitions.yaml"
SUPPORTED_TEMPLATE_FORMATS = frozenset({"txt", "md", "docx", "pdf"})


class WritingTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    formats: list[str] = Field(..., min_length=1)
    tags: list[str]
    required_fields: list[str]
    template_file: str = Field(..., min_length=1)
    default_title: str | None = None
    default_filename: str | None = None
    version: int = Field(..., ge=1)

    @field_validator("id", "name", "description", "category", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("formats", "tags", "required_fields", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [item.strip().lower() if isinstance(item, str) else item for item in value]

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        unsupported = sorted(set(value) - SUPPORTED_TEMPLATE_FORMATS)
        if unsupported:
            raise ValueError(
                "Unsupported template format(s): "
                + ", ".join(unsupported)
                + ". Supported formats are txt, md, docx, and pdf."
            )
        return list(dict.fromkeys(value))


class WritingTemplateMetadata(BaseModel):
    id: str
    name: str
    description: str
    category: str
    formats: list[str]
    tags: list[str]
    required_fields: list[str]
    default_title: str | None = None
    default_filename: str | None = None
    version: int

    @classmethod
    def from_template(cls, template: WritingTemplate) -> "WritingTemplateMetadata":
        """Project a validated definition into the metadata-only result model."""
        return cls(**template.model_dump(exclude={"template_file"}))


class WritingTemplateListResult(BaseModel):
    templates: list[WritingTemplateMetadata] = Field(default_factory=list)


class WritingTemplatesArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="Reserved for database-backed template search; currently ignored.",
    )
    category: str | None = Field(default=None, description="Optional template category filter.")
    format: str | None = Field(
        default=None,
        description="Optional output format filter: txt, md, docx, or pdf.",
    )


def load_writing_templates(directory: Path | None = None) -> list[WritingTemplate]:
    directory = TEMPLATES_DIR if directory is None else directory
    if not directory.is_dir():
        raise FileNotFoundError(f"Writing templates directory does not exist: {directory}")

    definitions_path = directory / TEMPLATE_DEFINITIONS_FILE
    if not definitions_path.is_file():
        raise FileNotFoundError(f"Writing template definitions file does not exist: {definitions_path}")
    try:
        with definitions_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in writing template definitions: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read writing template definitions: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("templates"), list):
        raise ValueError(
            f"Writing template definitions '{definitions_path.name}' must contain a 'templates' list."
        )

    templates: list[WritingTemplate] = []
    seen_ids: dict[str, Path] = {}
    for index, definition in enumerate(payload["templates"]):
        definition_name = f"{definitions_path.name} entry {index + 1}"
        if not isinstance(definition, dict):
            raise ValueError(f"Writing template definition '{definition_name}' must be a YAML mapping.")
        try:
            template = WritingTemplate.model_validate(definition)
        except ValidationError as exc:
            raise ValueError(f"Invalid writing template definition '{definition_name}': {exc}") from exc
        template_path = (directory / template.template_file).resolve()
        if template_path.parent != directory.resolve() or not template_path.is_file():
            raise ValueError(
                f"Writing template '{template.id}' references missing or unsafe template file "
                f"'{template.template_file}'."
            )
        previous_entry = seen_ids.get(template.id)
        if previous_entry is not None:
            raise ValueError(
                f"Duplicate writing template id '{template.id}' in "
                f"'{previous_entry.name}' and '{definitions_path.name}'."
            )
        seen_ids[template.id] = definitions_path
        templates.append(template)
    return templates


def load_writing_template_content(
    template_id: str,
    directory: Path | None = None,
) -> str:
    """Lazily load one template body after its definition has been selected."""
    directory = TEMPLATES_DIR if directory is None else directory
    template = next(
        (item for item in load_writing_templates(directory) if item.id == template_id),
        None,
    )
    if template is None:
        raise ValueError(f"Unknown writing template '{template_id}'.")
    template_path = (directory / template.template_file).resolve()
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read writing template '{template.template_file}'.") from exc


def _matches(template: WritingTemplate, args: WritingTemplatesArgs) -> bool:
    if args.category and template.category.casefold() != args.category.strip().casefold():
        return False
    if args.format and args.format.strip().lower() not in template.formats:
        return False
    # Query matching is intentionally deferred until template metadata is
    # stored in the database. For now, discovery returns the catalog entries
    # selected by the structured category/format filters.
    return True


@tool(
    TOOL_NAME_WRITING_TEMPLATES,
    args_schema=WritingTemplatesArgs,
    description=(
        "Discover available writing templates and their supported output formats. "
        "Returns template metadata only; it does not return template content."
    ),
)
def writing_templates(
    query: str | None = None,
    category: str | None = None,
    format: str | None = None,
) -> ToolResult:
    args = WritingTemplatesArgs(query=query, category=category, format=format)
    if args.format and args.format.strip().lower() not in SUPPORTED_TEMPLATE_FORMATS:
        raise ValueError(
            f"Unsupported template format '{args.format}'. "
            "Supported formats are txt, md, docx, and pdf."
        )
    matches = [template for template in load_writing_templates() if _matches(template, args)]
    metadata = [WritingTemplateMetadata.from_template(template) for template in matches]
    result = WritingTemplateListResult(templates=metadata)
    evidence = [
        EvidenceView(
            item_id=template.id,
            tool_name=TOOL_NAME_WRITING_TEMPLATES,
            title=template.name,
            summary=template.description,
            source=TOOL_NAME_WRITING_TEMPLATES,
            entity_type=TOOL_RESULT_TYPE_WRITING_TEMPLATES,
            llm_metadata=WritingTemplateMetadata.from_template(template).model_dump(mode="json"),
        )
        for template in matches
    ]
    return ToolResult(result=result, evidence=evidence)


__all__ = [
    "TEMPLATES_DIR",
    "TEMPLATE_DEFINITIONS_FILE",
    "WritingTemplate",
    "WritingTemplateMetadata",
    "WritingTemplateListResult",
    "load_writing_templates",
    "load_writing_template_content",
    "writing_templates",
]
