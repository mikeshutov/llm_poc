from __future__ import annotations

from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.evidence import EvidenceBundle, EvidenceView, ToolResult
from tool.tools import get_tool_result_type


def _resolve_step_id(
    tool_result: ToolResult,
    *,
    fallback_step_id: str,
) -> str:
    return tool_result.step_id.strip() or fallback_step_id


def _resolve_tool_name(
    tool_result: ToolResult,
    *,
    fallback_tool_name: str,
) -> str:
    if tool_result.tool_name.strip():
        return tool_result.tool_name.strip()
    for evidence in tool_result.evidence:
        if evidence.tool_name.strip():
            return evidence.tool_name.strip()
        if evidence.source.strip():
            return evidence.source.strip()
    return fallback_tool_name


def build_evidence_bundle_from_tool_results(tool_results: list[ToolResult]) -> EvidenceBundle:
    hydrated_evidence_by_id: dict[str, EvidenceView] = {}
    evidence_views_by_step_id: dict[str, list[EvidenceView]] = {}

    for tool_result in tool_results:
        resolved_step_id = _resolve_step_id(tool_result, fallback_step_id="")
        if not resolved_step_id:
            continue
        resolved_tool_name = _resolve_tool_name(tool_result, fallback_tool_name="")
        evidence, evidence_views = _rehydrate_tool_result_records(
            tool_result,
            step_id=resolved_step_id,
            tool_name=resolved_tool_name,
        )
        if not evidence:
            continue
        evidence_views_by_step_id[resolved_step_id] = evidence_views
        for evidence in evidence:
            hydrated_evidence_by_id[str(evidence.id)] = evidence

    return EvidenceBundle(
        hydrated_evidence_by_id=hydrated_evidence_by_id,
        evidence_views_by_step_id=evidence_views_by_step_id,
    )


def build_evidence_steps_from_tool_results(
    tool_results: list[ToolResult],
    evidence_views_by_step_id: dict[str, list[EvidenceView]],
) -> list[EvidenceStep]:
    evidence_steps: list[EvidenceStep] = []
    evidence_step_by_type: dict[str, EvidenceStep] = {}
    for tool_result in tool_results:
        resolved_step_id = _resolve_step_id(tool_result, fallback_step_id="")
        if not resolved_step_id:
            continue
        resolved_tool_name = _resolve_tool_name(tool_result, fallback_tool_name="")
        step_type = get_tool_result_type(resolved_tool_name)
        step_metadata = dict(tool_result.metadata)
        step_evidence = list(evidence_views_by_step_id.get(resolved_step_id, []))
        existing_step = evidence_step_by_type.get(step_type)
        if existing_step is None:
            evidence_step = EvidenceStep(
                type=step_type,
                metadata=step_metadata,
                evidence=step_evidence,
            )
            evidence_step_by_type[step_type] = evidence_step
            evidence_steps.append(evidence_step)
            continue
        existing_step.metadata = _merge_step_metadata(existing_step.metadata, step_metadata)
        existing_step.evidence.extend(step_evidence)
    return evidence_steps


def filter_evidence_steps(
    evidence_steps: list[EvidenceStep],
    relevant_evidence_ids: set[str],
) -> list[EvidenceStep]:
    if not relevant_evidence_ids:
        return evidence_steps

    filtered_steps: list[EvidenceStep] = []
    for step in evidence_steps:
        matching_evidence = [
            evidence
            for evidence in step.evidence
            if str(evidence.id) in relevant_evidence_ids
        ]
        if not matching_evidence:
            continue
        filtered_steps.append(
            EvidenceStep(
                type=step.type,
                metadata=dict(step.metadata),
                evidence=matching_evidence,
            )
        )
    return filtered_steps


def _rehydrate_tool_result_records(
    tool_result: ToolResult,
    *,
    step_id: str,
    tool_name: str,
) -> tuple[list[EvidenceView], list[EvidenceView]]:
    evidence = [
        _rehydrate_evidence_item(
            evidence,
            step_id=step_id,
            tool_name=tool_name,
        )
        for evidence in tool_result.evidence
    ]
    return evidence, list(evidence)


def _rehydrate_evidence_item(
    evidence: EvidenceView,
    *,
    step_id: str,
    tool_name: str,
) -> EvidenceView:
    merged = evidence.model_copy(deep=True)
    merged.step_id = step_id
    if not merged.tool_name:
        merged.tool_name = tool_name
    if not merged.source:
        merged.source = tool_name
    if not merged.entity_type:
        merged.entity_type = get_tool_result_type(tool_name)
    return merged


def _merge_step_metadata(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    if not left:
        return dict(right)
    if not right:
        return dict(left)

    merged = dict(left)
    for key, value in right.items():
        if key not in merged:
            merged[key] = value
            continue
        if merged[key] == value:
            continue
        merged[key] = _merge_metadata_values(merged[key], value)
    return merged


def _merge_metadata_values(left: object, right: object) -> object:
    left_values = left if isinstance(left, list) else [left]
    right_values = right if isinstance(right, list) else [right]
    merged_values: list[object] = []
    for value in [*left_values, *right_values]:
        if value in merged_values:
            continue
        merged_values.append(value)
    return merged_values

