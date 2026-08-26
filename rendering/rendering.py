import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import streamlit as st

from common.config import CONTENT_KEY, FILES_DIR, IMAGE_MIME_PREFIX, ROLE_ASSISTANT, ROLE_DEBUG, ROLE_KEY
from common.logging import fetch_agent_logs_for_roundtrip, fetch_llm_call_payloads_for_roundtrip
from llm.usage import build_llm_usage_payload
from rendering.cards import render_cards, render_magic_card_evidence_cards, render_magic_card_rulings
from rendering.debug import debug_render_message, render_agent_logs
from rendering.feedback import render_feedback_controls
from rendering.replay import render_replay_control
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from tool.constants import TOOL_NAME_GENERIC_WEB_SEARCH
from tool.constants import TOOL_NAME_GET_COMMANDER_CARDS
from tool.constants import TOOL_NAME_SEARCH_MAGIC_CARDS
from tool.constants import TOOL_NAME_STRUCTURED_FACTS_LOOKUP
from tool.constants import TOOL_NAME_WIKIPEDIA_SEARCH
from tool.constants import TOOL_RESULT_TYPE_CARD_RESULTS
from tool.constants import TOOL_RESULT_TYPE_MEAL_RESULTS
from tool.constants import TOOL_RESULT_TYPE_RULES
from tool.constants import TOOL_RESULT_TYPE_WEATHER


@dataclass(frozen=True)
class InlineEvidenceReference:
    evidence_id: str
    title: str
    source: str


@dataclass(frozen=True)
class EvidenceCard:
    id: str
    name: str
    description: str
    url: str
    image_url: str
    source: str
    entity_type: str


CARD_LAYOUTS_BY_EVIDENCE_TYPE: dict[str, dict[str, int]] = {
    TOOL_RESULT_TYPE_MEAL_RESULTS: {"per_row": 2},
}


INLINE_EVIDENCE_TYPES: set[str] = {
    TOOL_RESULT_TYPE_WEATHER,
}

INLINE_EVIDENCE_TOOL_NAMES: set[str] = {
    TOOL_NAME_GENERIC_WEB_SEARCH,
    TOOL_NAME_WIKIPEDIA_SEARCH,
    TOOL_NAME_STRUCTURED_FACTS_LOOKUP,
}


def _is_inline_link_evidence(evidence: EvidenceView) -> bool:
    return (
        evidence.tool_name.strip() in INLINE_EVIDENCE_TOOL_NAMES
        or evidence.source.strip() in INLINE_EVIDENCE_TOOL_NAMES
    )


def _is_inline_label_evidence(evidence: EvidenceView) -> bool:
    return evidence.entity_type.strip() in INLINE_EVIDENCE_TYPES


def _is_inline_evidence(evidence: EvidenceView) -> bool:
    return _is_inline_link_evidence(evidence) or _is_inline_label_evidence(evidence)


def _is_magic_card_evidence(evidence: EvidenceView) -> bool:
    return (
        evidence.tool_name.strip() in {TOOL_NAME_SEARCH_MAGIC_CARDS, TOOL_NAME_GET_COMMANDER_CARDS}
        and evidence.entity_type.strip() == TOOL_RESULT_TYPE_CARD_RESULTS
    )


def _is_magic_card_ruling_evidence(evidence: EvidenceView) -> bool:
    return evidence.entity_type.strip() == TOOL_RESULT_TYPE_RULES


def format_timestamp(ts) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    if isinstance(ts, datetime):
        local = ts.astimezone()
        hour = local.strftime("%I").lstrip("0") or "12"
        day = local.strftime("%d").lstrip("0") or "1"
        return f"{hour}:{local.strftime('%M %p')} | {local.strftime('%b')} {day}, {local.strftime('%Y')}"
    return None


def _format_roundtrip_usage_summary(llm_usage: dict | None) -> str | None:
    if not isinstance(llm_usage, dict):
        return None
    summary = llm_usage.get("summary")
    if not isinstance(summary, dict):
        return None

    total_tokens = summary.get("total_tokens")
    total_cost = summary.get("computed_total_cost")
    if total_tokens is None and total_cost is None:
        return None

    parts: list[str] = []
    if total_tokens is not None:
        parts.append(f"Total Tokens: {int(total_tokens):,}")
    if total_cost is not None:
        parts.append(f"Estimated Cost: ${total_cost}")
    return " | ".join(parts) if parts else None


def _format_roundtrip_duration(payload: dict | None) -> str | None:
    if isinstance(payload, dict):
        latency_ms = payload.get("roundtrip_latency_ms")
        if latency_ms is not None:
            try:
                return f"{float(latency_ms) / 1000:.1f}s"
            except (TypeError, ValueError):
                return None
    return None


def _render_roundtrip_llm_usage(llm_usage: dict | None) -> None:
    if not isinstance(llm_usage, dict):
        return
    summary = llm_usage.get("summary") if isinstance(llm_usage.get("summary"), dict) else {}
    calls = llm_usage.get("calls") if isinstance(llm_usage.get("calls"), list) else []
    if not summary and not calls:
        return

    title = f"LLM Usage ({llm_usage.get('retrieved_call_count', len(calls))} calls)"
    with st.expander(title):
        if summary:
            st.json(summary, expanded=False)
        if calls:
            st.json(calls, expanded=False)


def fetch_llm_usage_for_roundtrip(roundtrip_id: str | None) -> dict[str, Any] | None:
    llm_calls = fetch_llm_call_payloads_for_roundtrip(roundtrip_id)
    if not llm_calls:
        return None
    return build_llm_usage_payload(llm_calls)


def _get_evidence_by_id(payload: dict | None) -> dict[str, EvidenceView]:
    if not isinstance(payload, dict):
        return {}
    raw_evidence_by_id = payload.get("evidence_by_id")
    if not isinstance(raw_evidence_by_id, dict):
        return {}
    evidence_by_id: dict[str, EvidenceView] = {}
    for evidence_id, evidence in raw_evidence_by_id.items():
        if not isinstance(evidence_id, str) or not isinstance(evidence, dict):
            continue
        try:
            evidence_by_id[evidence_id] = EvidenceView.model_validate(evidence)
        except Exception:
            continue
    return evidence_by_id


def _normalize_block_evidence_ids(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceView],
) -> list[str]:
    if not evidence_ids or not evidence_by_id:
        return evidence_ids

    normalized_ids: list[str] = []
    for evidence_id in evidence_ids:
        normalized = evidence_id.strip()
        if not normalized:
            continue
        if normalized in evidence_by_id:
            normalized_ids.append(normalized)
            continue
        normalized_ids.append(normalized)
    return normalized_ids


def get_renderable_result_blocks(content: str, payload: dict | None) -> list[SynthesisResultBlock]:
    evidence_by_id = _get_evidence_by_id(payload)
    if isinstance(payload, dict):
        raw_blocks = payload.get("result")
        if isinstance(raw_blocks, list):
            blocks: list[SynthesisResultBlock] = []
            for raw_block in raw_blocks:
                if not isinstance(raw_block, dict):
                    continue
                block_content = raw_block.get("content")
                if not isinstance(block_content, str):
                    continue
                content_text = block_content.strip()
                if not content_text:
                    continue
                raw_evidence_ids = raw_block.get("evidence_ids", [])
                evidence_ids = []
                if isinstance(raw_evidence_ids, list):
                    evidence_ids = [
                        evidence_id.strip()
                        for evidence_id in raw_evidence_ids
                        if isinstance(evidence_id, str) and evidence_id.strip()
                    ]
                evidence_ids = _normalize_block_evidence_ids(
                    evidence_ids,
                    evidence_by_id,
                )
                blocks.append(
                    SynthesisResultBlock(
                        content=content_text,
                        evidence_ids=evidence_ids,
                    )
                )
            if blocks:
                return blocks
    trimmed_content = content.strip()
    if not trimmed_content:
        return []
    return [SynthesisResultBlock(content=trimmed_content, evidence_ids=[])]


def _build_inline_evidence(
    block: SynthesisResultBlock,
    evidence_by_id: dict[str, EvidenceView],
) -> list[InlineEvidenceReference]:
    inline_evidence: list[InlineEvidenceReference] = []
    for evidence_id in block.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        if not _is_inline_evidence(evidence):
            continue
        inline_evidence.append(
            InlineEvidenceReference(
                evidence_id=evidence_id,
                title=evidence.title,
                source=evidence.source,
            )
        )
    return inline_evidence


def _build_block_cards(
    block: SynthesisResultBlock,
    evidence_by_id: dict[str, EvidenceView],
) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for evidence_id in block.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            continue
        if _is_inline_evidence(evidence) or _is_magic_card_evidence(evidence) or _is_magic_card_ruling_evidence(evidence):
            continue
        url = _primary_card_url(evidence.urls)
        image_url = evidence.image_url.strip()
        title = evidence.title.strip()
        summary = evidence.summary.strip()
        if not title or not (url or image_url):
            continue
        cards.append(
            EvidenceCard(
                id=evidence_id,
                name=title,
                description=summary,
                url=url,
                image_url=image_url,
                source=evidence.source.strip(),
                entity_type=evidence.entity_type.strip(),
            )
        )
    return cards


def _primary_card_url(urls: list[EvidenceUrl]) -> str:
    for preferred_type in ("website", "youtube"):
        for entry in urls:
            cleaned_url = entry.url.strip()
            if entry.url_type == preferred_type and cleaned_url:
                return cleaned_url
    return ""


def _render_result_content(content: str, links: list[tuple[str, str]]) -> None:
    chips = "".join(
        (
            "<a "
            f'href="{html.escape(url, quote=True)}" '
            'target="_blank" rel="noopener noreferrer" '
            "style=\"display:inline-flex;align-items:center;gap:0.2rem;margin:0 0.25rem 0 0.35rem;"
            "padding:0.08rem 0.4rem;border:1px solid rgba(49,51,63,0.22);border-radius:999px;"
            "font-size:0.72rem;line-height:1.25;text-decoration:none;color:inherit;\">"
            f"<span aria-hidden=\"true\">&#8599;</span> {html.escape(label)}</a>"
        )
        for label, url in links
    )
    escaped_content = html.escape(content).replace("\n", "<br>")
    st.html(f'<span style="white-space:pre-wrap">{escaped_content}</span>{chips}')


def _render_result_block(
    block: SynthesisResultBlock,
    evidence_by_id: dict[str, EvidenceView],
) -> list[EvidenceCard]:
    inline_evidence = _build_inline_evidence(block, evidence_by_id)
    inline_links: list[tuple[str, str]] = []
    inline_labels: list[str] = []
    for evidence in inline_evidence:
        evidence = evidence_by_id.get(evidence.evidence_id)
        if evidence is None:
            continue
        evidence_url = _primary_card_url(evidence.urls)
        if evidence_url:
            label = evidence.title.strip() or evidence.source.strip() or evidence.evidence_id
            inline_links.append((label, evidence_url))
            continue
        label = evidence.summary.strip() or evidence.title.strip() or evidence.evidence_id
        source = evidence.source.strip()
        inline_labels.append(f"{label} ({source})" if source else label)

    _render_result_content(block.content, inline_links)

    magic_card_evidence = [
        evidence
        for evidence_id in block.evidence_ids
        if (evidence := evidence_by_id.get(evidence_id)) is not None and _is_magic_card_evidence(evidence)
    ]
    if magic_card_evidence:
        render_magic_card_evidence_cards(magic_card_evidence)

    magic_card_rulings = [
        evidence
        for evidence_id in block.evidence_ids
        if (evidence := evidence_by_id.get(evidence_id)) is not None and _is_magic_card_ruling_evidence(evidence)
    ]
    if magic_card_rulings:
        render_magic_card_rulings(magic_card_rulings)

    block_cards = _build_block_cards(block, evidence_by_id)
    for label in inline_labels:
        st.caption(label)
    return block_cards


def render_assistant_content(
    content: str,
    payload: dict | None,
    *,
    roundtrip_id: str | None = None,
    assistant_follow_up: str | None = None,
) -> None:
    next_question = assistant_follow_up
    if next_question is None and isinstance(payload, dict):
        next_question = payload.get("next_question")
    evidence_by_id = _get_evidence_by_id(payload)
    result_blocks = get_renderable_result_blocks(content, payload)
    has_next_question = isinstance(next_question, str) and bool(next_question)
    all_block_cards: list[EvidenceCard] = []
    llm_usage = fetch_llm_usage_for_roundtrip(roundtrip_id)

    for block in result_blocks:
        all_block_cards.extend(_render_result_block(block, evidence_by_id))

    if all_block_cards:
        deduped_cards: list[EvidenceCard] = []
        seen_card_ids: set[str] = set()
        for card in all_block_cards:
            if card.id in seen_card_ids:
                continue
            seen_card_ids.add(card.id)
            deduped_cards.append(card)
        cards_by_layout: dict[int, list[EvidenceCard]] = {}
        for card in deduped_cards:
            layout = CARD_LAYOUTS_BY_EVIDENCE_TYPE.get(card.entity_type, {})
            per_row = layout.get("per_row", 3)
            cards_by_layout.setdefault(per_row, []).append(card)

        for per_row, cards in cards_by_layout.items():
            render_cards(
                [card.__dict__ for card in cards],
                per_row=per_row,
                heading_key="name",
                description_key="description",
                image_key="image_url",
                link_key="url",
            )

    if has_next_question:
        st.markdown(next_question)

    _render_roundtrip_llm_usage(llm_usage)
    render_agent_logs(fetch_agent_logs_for_roundtrip(roundtrip_id))


def _render_file_preview(attached_file: dict) -> None:
    name = attached_file.get("name", "")
    mime = attached_file.get("type", "")
    path = os.path.join(FILES_DIR, name)
    if mime.startswith(IMAGE_MIME_PREFIX) and os.path.exists(path):
        st.image(path, width=200)
    else:
        icon = "[PDF]" if "pdf" in mime else "[FILE]"
        st.markdown(
            f"<span style='background:#f0f2f6;border-radius:6px;padding:3px 10px;font-size:0.85em'>{icon} {name}</span>",
            unsafe_allow_html=True,
        )


def render_message(msg: dict) -> None:
    role = msg[ROLE_KEY]
    content = msg[CONTENT_KEY]
    content_title = msg.get("title", "Debug")
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else None
    timestamp = format_timestamp(msg.get("timestamp"))
    duration = _format_roundtrip_duration(payload)
    footer_timestamp = " | ".join(part for part in [duration, timestamp] if part) or None
    if msg.get("status"):
        with st.chat_message("assistant", avatar=":material/more_horiz:"):
            st.markdown(content)
    elif role == ROLE_DEBUG:
        debug_render_message(content, content_title)
    else:
        with st.chat_message(role):
            if role == ROLE_ASSISTANT:
                render_assistant_content(
                    content,
                    msg.get("payload"),
                    roundtrip_id=msg.get("roundtrip_id"),
                    assistant_follow_up=msg.get("assistant_follow_up"),
                )
                render_feedback_controls(
                    roundtrip_id=msg.get("roundtrip_id"),
                    model=msg.get("model"),
                    sources_payload=payload,
                    feedback_id=msg.get("feedback_id"),
                    timestamp=footer_timestamp,
                    usage_summary=_format_roundtrip_usage_summary(
                        fetch_llm_usage_for_roundtrip(msg.get("roundtrip_id"))
                    ),
                )
            else:
                st.write(content)
                attached_file = msg.get("attached_file")
                if attached_file:
                    _render_file_preview(attached_file)
                render_replay_control(
                    roundtrip_id=msg.get("roundtrip_id"),
                    timestamp=timestamp,
                )
