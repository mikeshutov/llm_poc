from __future__ import annotations

from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import IterationState
from request_orchestrator.models.evidence import EvidenceBundle, EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.models.plan_step_ids import format_plan_step_id
from tool.tools import get_tool_result_type


def build_evidence_bundle(iteration_trace: list[IterationState]) -> EvidenceBundle:
    hydrated_evidence_by_id: dict[str, HydratedEvidence] = {}
    evidence_views_by_step_id: dict[str, list[EvidenceView]] = {}

    for iteration_number, iteration in enumerate(iteration_trace, start=1):
        if iteration.plan is None:
            continue
        for step in iteration.plan.steps:
            step_id = format_plan_step_id(iteration_number, step.id)
            tool_result = iteration.results.get(step_id)
            if not isinstance(tool_result, ToolResult):
                continue
            hydrated_evidence, evidence_views = _rehydrate_tool_result_records(
                tool_result,
                step_id=step_id,
                tool_name=step.tool,
            )
            if not hydrated_evidence:
                continue
            evidence_views_by_step_id[step_id] = evidence_views
            for evidence in hydrated_evidence:
                hydrated_evidence_by_id[evidence.evidence_id] = evidence

    return EvidenceBundle(
        hydrated_evidence_by_id=hydrated_evidence_by_id,
        evidence_views_by_step_id=evidence_views_by_step_id,
    )


def build_evidence_steps(
    iteration_trace: list[IterationState],
    evidence_views_by_step_id: dict[str, list[EvidenceView]],
) -> list[EvidenceStep]:
    evidence_steps: list[EvidenceStep] = []
    for iteration_number, iteration in enumerate(iteration_trace, start=1):
        if iteration.plan is None:
            continue
        for step in iteration.plan.steps:
            step_id = format_plan_step_id(iteration_number, step.id)
            tool_result = iteration.results.get(step_id)
            step_type = get_tool_result_type(step.tool)
            evidence_step = EvidenceStep(
                type=step_type,
                metadata=dict(tool_result.metadata) if isinstance(tool_result, ToolResult) else {},
                evidence=list(evidence_views_by_step_id.get(step_id, [])),
            )
            evidence_steps.append(evidence_step)
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
            if evidence.evidence_id in relevant_evidence_ids
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
) -> tuple[list[HydratedEvidence], list[EvidenceView]]:
    hydrated_evidence = [
        _rehydrate_evidence_item(
            evidence,
            step_id=step_id,
            tool_name=tool_name,
            reference_id=index,
        )
        for index, evidence in enumerate(tool_result.hydrated_evidence, start=1)
    ]
    if tool_result.evidence_views:
        evidence_views = [
            _rehydrate_evidence_view_item(
                evidence_view,
                hydrated_evidence_item=hydrated_evidence[index] if index < len(hydrated_evidence) else None,
                step_id=step_id,
                reference_id=index + 1,
            )
            for index, evidence_view in enumerate(tool_result.evidence_views)
        ]
        if not hydrated_evidence:
            hydrated_evidence = [
                _build_hydrated_evidence_from_view(
                    evidence_view,
                    step_id=step_id,
                    tool_name=tool_name,
                    reference_id=index,
                )
                for index, evidence_view in enumerate(evidence_views, start=1)
            ]
    else:
        evidence_views = [build_evidence_view(evidence) for evidence in hydrated_evidence]
    return hydrated_evidence, evidence_views


def _rehydrate_evidence_item(
    evidence: HydratedEvidence,
    *,
    step_id: str,
    tool_name: str,
    reference_id: int,
) -> HydratedEvidence:
    merged = evidence.model_copy(deep=True)
    merged.step_id = step_id
    merged.evidence_id = _format_evidence_id(step_id, reference_id)
    if not merged.tool_name:
        merged.tool_name = tool_name
    if not merged.source:
        merged.source = tool_name
    if not merged.entity_type:
        merged.entity_type = get_tool_result_type(tool_name)
    return merged


def _rehydrate_evidence_view_item(
    evidence_view: EvidenceView,
    *,
    hydrated_evidence_item: HydratedEvidence | None,
    step_id: str,
    reference_id: int,
) -> EvidenceView:
    merged = evidence_view.model_copy(deep=True)
    merged.evidence_id = _format_evidence_id(step_id, reference_id)
    if not merged.item_id and hydrated_evidence_item is not None:
        merged.item_id = hydrated_evidence_item.item_id
    return merged


def build_evidence_view(evidence: HydratedEvidence) -> EvidenceView:
    return EvidenceView(
        evidence_id=evidence.evidence_id,
        item_id=evidence.item_id,
        title=evidence.title,
        summary=evidence.summary,
        metadata=dict(evidence.metadata),
    )


def _build_hydrated_evidence_from_view(
    evidence_view: EvidenceView,
    *,
    step_id: str,
    tool_name: str,
    reference_id: int,
) -> HydratedEvidence:
    return HydratedEvidence(
        evidence_id=evidence_view.evidence_id or _format_evidence_id(step_id, reference_id),
        step_id=step_id,
        item_id=evidence_view.item_id,
        tool_name=tool_name,
        title=evidence_view.title,
        summary=evidence_view.summary,
        source=tool_name,
        entity_type=get_tool_result_type(tool_name),
        metadata=dict(evidence_view.metadata),
        evidence_object=evidence_view.evidence_object,
    )

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


def _format_evidence_id(step_id: str, reference_id: int) -> str:
    return f"{step_id}R{reference_id}"
