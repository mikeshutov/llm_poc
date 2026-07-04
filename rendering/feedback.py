from __future__ import annotations

from dataclasses import asdict
from uuid import UUID

import streamlit as st

from conversation.models.conversation_models import RoundtripFeedback
from conversation.repository.conversation_repository import ConversationRepository

FEEDBACK_TARGET_KEY = "feedback_target"
FEEDBACK_DIALOG_INIT_KEY = "feedback_dialog_initialized_for"


def clear_feedback_state() -> None:
    target = st.session_state.pop(FEEDBACK_TARGET_KEY, None)
    st.session_state.pop(FEEDBACK_DIALOG_INIT_KEY, None)
    if isinstance(target, dict):
        roundtrip_id = target.get("roundtrip_id")
        if roundtrip_id:
            for suffix in ("met_expectation", "reason", "expected_answer"):
                st.session_state.pop(f"feedback_{suffix}_{roundtrip_id}", None)


def request_feedback_dialog(roundtrip_id: str, met_expectation: bool, model: str | None = None) -> None:
    current = st.session_state.get(FEEDBACK_TARGET_KEY)
    if isinstance(current, dict) and current.get("roundtrip_id") != roundtrip_id:
        clear_feedback_state()
    st.session_state[FEEDBACK_TARGET_KEY] = {
        "roundtrip_id": roundtrip_id,
        "met_expectation": met_expectation,
        "model": model,
    }
    st.session_state.pop(FEEDBACK_DIALOG_INIT_KEY, None)


def _widget_keys(roundtrip_id: str) -> tuple[str, str, str]:
    return (
        f"feedback_met_expectation_{roundtrip_id}",
        f"feedback_reason_{roundtrip_id}",
        f"feedback_expected_answer_{roundtrip_id}",
    )


def _initialize_dialog_state(conversation_repository: ConversationRepository, target: dict[str, object]) -> RoundtripFeedback | None:
    roundtrip_id = str(target["roundtrip_id"])
    if st.session_state.get(FEEDBACK_DIALOG_INIT_KEY) == roundtrip_id:
        existing = conversation_repository.get_roundtrip_feedback(UUID(roundtrip_id))
        return existing

    existing = conversation_repository.get_roundtrip_feedback(UUID(roundtrip_id))
    met_expectation_key, reason_key, expected_key = _widget_keys(roundtrip_id)
    st.session_state[met_expectation_key] = "Yes" if (existing.met_expectation if existing else bool(target.get("met_expectation", True))) else "No"
    st.session_state[reason_key] = existing.reason if existing and existing.reason is not None else ""
    st.session_state[expected_key] = existing.expected_answer if existing and existing.expected_answer is not None else ""
    st.session_state[FEEDBACK_DIALOG_INIT_KEY] = roundtrip_id
    return existing


def render_feedback_controls(
    *,
    roundtrip_id: UUID | str | None,
    model: str | None,
    feedback_id: UUID | None = None,
    timestamp: str | None = None,
) -> None:
    if roundtrip_id is None:
        return

    rid = str(roundtrip_id)
    col_caption, col_status, col_up, col_down = st.columns([8, 3, 1, 1], vertical_alignment="center")
    with col_caption:
        if timestamp:
            st.caption(timestamp)
    with col_status:
        if feedback_id is not None:
            st.caption("Feedback saved")
    with col_up:
        st.button(
            ":material/thumb_up:",
            key=f"feedback_up_{rid}",
            help="Thumbs up",
            on_click=request_feedback_dialog,
            args=(rid, True, model),
        )
    with col_down:
        st.button(
            ":material/thumb_down:",
            key=f"feedback_down_{rid}",
            help="Thumbs down",
            on_click=request_feedback_dialog,
            args=(rid, False, model),
        )


@st.dialog("Message feedback")
def render_feedback_dialog(conversation_repository: ConversationRepository) -> None:
    target = st.session_state.get(FEEDBACK_TARGET_KEY)
    if not isinstance(target, dict):
        return

    roundtrip_id = str(target["roundtrip_id"])
    existing = _initialize_dialog_state(conversation_repository, target)
    met_expectation_key, reason_key, expected_key = _widget_keys(roundtrip_id)

    st.write("Tell us what worked well or what was missing.")
    if existing:
        st.caption(f"Current saved feedback: {'met expectation' if existing.met_expectation else 'did not meet expectation'}")

    with st.form(f"feedback_form_{roundtrip_id}"):
        met_expectation = st.radio(
            "Met expectation",
            options=("Yes", "No"),
            horizontal=True,
            key=met_expectation_key,
        )
        reason = st.text_area(
            "Reason",
            key=reason_key,
            placeholder="Optional: tell us why it did or did not meet expectation.",
        )
        expected_answer = st.text_area(
            "Expected answer",
            key=expected_key,
            placeholder="Optional: what answer were you expecting?",
        )
        submit_col, cancel_col = st.columns(2)
        with submit_col:
            save = st.form_submit_button("Save feedback", type="primary")
        with cancel_col:
            cancel = st.form_submit_button("Cancel")

    if save:
        saved = conversation_repository.upsert_roundtrip_feedback(
            UUID(roundtrip_id),
            met_expectation=met_expectation == "Yes",
            reason=reason.strip() or None,
            expected_answer=expected_answer.strip() or None,
            model=str(target.get("model")) if target.get("model") is not None else None,
        )
        for message in st.session_state.get("messages", []):
            if isinstance(message, dict) and message.get("roundtrip_id") == roundtrip_id:
                message["feedback_id"] = str(saved.id)
                break
        clear_feedback_state()
        st.rerun()
    if cancel:
        clear_feedback_state()
        st.rerun()
