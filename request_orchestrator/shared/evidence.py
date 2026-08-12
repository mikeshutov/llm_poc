from __future__ import annotations

from typing import Any

from common.data import prune_empty_prompt_values, sanitize_for_json_storage
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import IterationState
from request_orchestrator.models.evidence import EvidenceBundle, EvidenceUrl, EvidenceView, HydratedEvidence
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
            raw_output = iteration.results.get(step_id)
            hydrated_evidence = build_hydrated_evidence_for_output(
                raw_output,
                step_id=step_id,
                tool_name=step.tool,
            )
            if not hydrated_evidence:
                continue
            evidence_views_by_step_id[step_id] = [
                build_evidence_view(evidence)
                for evidence in hydrated_evidence
            ]
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
            evidence_steps.append(
                EvidenceStep(
                    type=get_tool_result_type(step.tool),
                    evidence=list(evidence_views_by_step_id.get(step_id, [])),
                )
            )
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
                evidence=matching_evidence,
            )
        )
    return filtered_steps


def build_hydrated_evidence_for_output(
    raw_output: Any,
    *,
    step_id: str,
    tool_name: str,
) -> list[HydratedEvidence]:
    if raw_output is None:
        return []

    payload = prune_empty_prompt_values(sanitize_for_json_storage(raw_output))
    items = _extract_evidence_items(payload)
    if items is not None:
        return [
            _build_generic_hydrated_evidence(
                item,
                step_id=step_id,
                tool_name=tool_name,
                reference_id=index,
            )
            for index, item in enumerate(items, start=1)
        ]

    return [
        _build_generic_hydrated_evidence(
            payload,
            step_id=step_id,
            tool_name=tool_name,
            reference_id=1,
        )
    ]


def build_evidence_view(evidence: HydratedEvidence) -> EvidenceView:
    return EvidenceView(
        evidence_id=evidence.evidence_id,
        item_id=evidence.item_id,
        title=evidence.title,
        summary=evidence.summary,
        metadata=dict(evidence.metadata),
    )


def _build_generic_hydrated_evidence(
    raw_output: Any,
    *,
    step_id: str,
    tool_name: str,
    reference_id: int,
) -> HydratedEvidence:
    payload = prune_empty_prompt_values(sanitize_for_json_storage(raw_output))
    item_id = _item_id_from_value(payload, fallback=reference_id)
    title = _generic_title(payload, tool_name=tool_name)
    summary = _generic_summary(payload, tool_name=tool_name)
    urls = _build_evidence_urls(payload)
    return HydratedEvidence(
        evidence_id=_format_evidence_id(step_id, reference_id),
        step_id=step_id,
        item_id=item_id,
        tool_name=tool_name,
        title=title,
        summary=summary,
        url=urls[0].url if urls else "",
        urls=urls,
        image_url=_first_non_empty(payload, "image_url", "image", "thumbnail") if isinstance(payload, dict) else "",
        published_at=_first_non_empty(payload, "published_at", "date", "time") if isinstance(payload, dict) else "",
        source=tool_name,
        entity_type="generic_result",
        location_name=_location_name_for_value(payload),
        metadata=_build_evidence_metadata(payload),
        raw_payload=payload,
    )


def _format_evidence_id(step_id: str, reference_id: int) -> str:
    return f"{step_id}R{reference_id}"


def _item_id_from_value(value: Any, *, fallback: int) -> str:
    if isinstance(value, dict):
        for text in (
            _first_non_empty(value, "id", "item_id", "entity_id", "qid", "url", "name", "title", "label", "itemLabel", "entityLabel"),
            _first_non_empty_nested(value, ("location", "name")),
        ):
            if text:
                return text
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            return text
    return str(fallback)

def _generic_title(value: Any, *, tool_name: str) -> str:
    if isinstance(value, dict):
        text = _first_non_empty(value, "title", "name", "label", "itemLabel", "entityLabel")
        if text:
            return text
    return tool_name.replace("_", " ").strip().title() or "Tool Result"


def _generic_summary(value: Any, *, tool_name: str) -> str:
    if isinstance(value, dict):
        structured_weather_summary = _structured_weather_summary(value)
        if structured_weather_summary:
            return structured_weather_summary
        text = _first_non_empty(
            value,
            "summary",
            "description",
            "snippet",
            "instructions",
            "content",
            "text",
        )
        if text:
            return text
        binding_summary = _flat_field_summary(value)
        if binding_summary:
            return binding_summary
        return (
            f"{tool_name} returned structured data with "
            f"{len(value)} top-level field{'s' if len(value) != 1 else ''}."
        )
    if isinstance(value, list):
        return f"{tool_name} returned {len(value)} item{'s' if len(value) != 1 else ''}."
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return f"{tool_name} returned no result."
    return f"{tool_name} returned a result."


def _extract_evidence_items(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("results", "bindings", "items", "meals", "internal_results", "external_results"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
    return None


def _flat_field_summary(value: dict[str, Any]) -> str:
    parts = [
        f"{key}={_stringify_scalar(item)}"
        for key, item in value.items()
        if item not in (None, "", [], {})
        and not isinstance(item, list)
        and _stringify_scalar(item)
    ]
    return ", ".join(parts)


def _structured_weather_summary(value: dict[str, Any]) -> str:
    location = value.get("location")
    weather = value.get("weather")
    if not isinstance(location, dict) or not isinstance(weather, dict):
        return ""

    location_name = _first_non_empty(location, "name", "city")
    country = _first_non_empty(location, "country")
    temperature = _first_non_empty(weather, "temperature")
    windspeed = _first_non_empty(weather, "windspeed")
    timestamp = _first_non_empty(weather, "time")

    parts: list[str] = []
    if temperature and location_name:
        location_text = location_name if not country else f"{location_name}, {country}"
        parts.append(f"{temperature} C in {location_text}")
    if windspeed:
        parts.append(f"wind {windspeed} km/h")
    if timestamp:
        parts.append(f"at {timestamp}")
    return ", ".join(parts)


def _location_name_for_value(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return _first_non_empty(
        value,
        "location_name",
        "city",
        "name",
    ) or _first_non_empty_nested(value, ("location", "name"))


def _build_evidence_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    ingredients = value.get("ingredients")
    if isinstance(ingredients, list):
        normalized_ingredients = [
            item
            for item in (
                _normalize_ingredient_entry(ingredient)
                for ingredient in ingredients
            )
            if item is not None
        ]
        if normalized_ingredients:
            return {"ingredients": normalized_ingredients}

    return {}


def _build_evidence_urls(value: Any) -> list[EvidenceUrl]:
    if not isinstance(value, dict):
        return []

    candidates = [
        ("website", _first_non_empty(value, "url", "link", "source")),
        ("youtube", _first_non_empty(value, "youtube")),
    ]

    seen_urls: set[str] = set()
    urls: list[EvidenceUrl] = []
    for url_type, url in candidates:
        cleaned_url = url.strip()
        if not cleaned_url or cleaned_url in seen_urls:
            continue
        seen_urls.add(cleaned_url)
        urls.append(EvidenceUrl(url=cleaned_url, url_type=url_type))
    return urls


def _first_non_empty(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        candidate = value.get(key)
        if candidate is None:
            continue
        text = _stringify_scalar(candidate)
        if text:
            return text
    return ""


def _first_non_empty_nested(value: dict[str, Any], *paths: tuple[str, ...]) -> str:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        text = _stringify_scalar(current)
        if text:
            return text
    return ""


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        nested_value = value.get("value")
        if nested_value is not None:
            return _stringify_scalar(nested_value)
        return ""
    if isinstance(value, list):
        return ""
    return str(value).strip()


def _normalize_ingredient_entry(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    name = _first_non_empty(value, "name")
    measure = _first_non_empty(value, "measure")
    if not name and not measure:
        return None

    entry: dict[str, str] = {}
    if name:
        entry["name"] = name
    if measure:
        entry["measure"] = measure
    return entry
