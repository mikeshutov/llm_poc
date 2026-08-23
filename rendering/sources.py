from __future__ import annotations

from typing import Any

import streamlit as st

from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView

SOURCES_DIALOG_KEY = "sources_dialog"


def clear_sources_panel() -> None:
    st.session_state.pop(SOURCES_DIALOG_KEY, None)


def request_sources_panel(roundtrip_id: str, payload: dict[str, Any]) -> None:
    st.session_state[SOURCES_DIALOG_KEY] = {
        "roundtrip_id": roundtrip_id,
        "payload": payload,
    }


def get_sources_panel_request() -> dict[str, Any] | None:
    payload = st.session_state.get(SOURCES_DIALOG_KEY)
    return payload if isinstance(payload, dict) else None


def render_sources_button(roundtrip_id: str | None, payload: dict[str, Any] | None) -> None:
    if roundtrip_id is None or not isinstance(payload, dict):
        return
    sources = _get_panel_sources(payload)
    if not sources:
        return
    st.button(
        f"Sources ({len(sources)})",
        key=f"sources_button_{roundtrip_id}",
        on_click=request_sources_panel,
        args=(roundtrip_id, payload),
    )


@st.dialog("Sources", width="large")
def render_sources_panel() -> None:
    request = get_sources_panel_request()
    if not request:
        return

    payload = request.get("payload")
    if not isinstance(payload, dict):
        clear_sources_panel()
        return

    sources = _get_panel_sources(payload)
    if not sources:
        clear_sources_panel()
        return

    for source in sources:
        label = source.title or source.evidence_id
        with st.expander(label, expanded=False):
            st.json(source.model_dump(), expanded=False)

    if st.button("Close", key="sources_dialog_close", use_container_width=True):
        clear_sources_panel()
        st.rerun()


def _get_panel_sources(payload: dict[str, Any]) -> list[EvidenceView]:
    hydrated_evidence_by_id = _get_hydrated_evidence_by_id(payload)
    if not hydrated_evidence_by_id:
        return []

    raw_used_evidence_ids = payload.get("used_evidence_ids")
    if isinstance(raw_used_evidence_ids, list):
        ordered_ids = [
            evidence_id
            for evidence_id in raw_used_evidence_ids
            if isinstance(evidence_id, str) and evidence_id in hydrated_evidence_by_id
        ]
        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        for evidence_id in ordered_ids:
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)
            deduped_ids.append(evidence_id)
        if deduped_ids:
            return [hydrated_evidence_by_id[evidence_id] for evidence_id in deduped_ids]

    return list(hydrated_evidence_by_id.values())


def _get_hydrated_evidence_by_id(payload: dict[str, Any]) -> dict[str, EvidenceView]:
    raw_hydrated_evidence_by_id = payload.get("hydrated_evidence_by_id")
    if not isinstance(raw_hydrated_evidence_by_id, dict):
        return {}
    hydrated_evidence_by_id: dict[str, EvidenceView] = {}
    for evidence_id, evidence in raw_hydrated_evidence_by_id.items():
        if not isinstance(evidence_id, str) or not isinstance(evidence, dict):
            continue
        try:
            hydrated_evidence_by_id[evidence_id] = EvidenceView.model_validate(evidence)
        except Exception:
            continue
    return hydrated_evidence_by_id


def _render_source_links(urls: list[EvidenceUrl]) -> None:
    deduped_links: list[EvidenceUrl] = []
    seen_urls: set[str] = set()
    for entry in urls:
        cleaned_url = entry.url.strip()
        if not cleaned_url or cleaned_url in seen_urls:
            continue
        seen_urls.add(cleaned_url)
        deduped_links.append(entry)

    for entry in deduped_links:
        label = entry.url_type.replace("_", " ").title()
        st.markdown(f"[{label}]({entry.url})")
