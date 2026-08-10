from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import streamlit as st

from common.model_constants import AVAILABLE_CHAT_MODELS
from conversation.models.conversation_model_config import CONVERSATION_MODEL_CONFIG_SPECS, ConversationModelConfig
from personalization.profile.repository.repo_factory import get_user_profile_repo
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from rendering.feedback import clear_feedback_state
from rendering.replay import clear_replay_state

MODEL_CONFIG_DIALOG_KEY = "conversation_model_config_dialog"
MODEL_CONFIG_SECTION_TITLES = {
    "main_agent": "Main Agent",
    "profile_agent": "Profile Management Agent",
    "shared": "Shared",
}
USER_CREATE_FORM_KEY = "show_create_user_form"
PROFILE_DETAILS_DIALOG_KEY = "profile_details_dialog"
PROFILE_EDIT_MODE_KEY = "profile_details_edit_mode"


def clear_conversation_model_config_dialog() -> None:
    st.session_state.pop(MODEL_CONFIG_DIALOG_KEY, None)


def request_conversation_model_config_dialog(
    conversation_id: str,
    title: str,
    replay_source_roundtrip_id: str | None = None,
) -> None:
    st.session_state[MODEL_CONFIG_DIALOG_KEY] = {
        "conversation_id": conversation_id,
        "title": title,
        "replay_source_roundtrip_id": replay_source_roundtrip_id,
    }


def get_conversation_model_config_dialog_request() -> dict[str, str] | None:
    payload = st.session_state.get(MODEL_CONFIG_DIALOG_KEY)
    return payload if isinstance(payload, dict) else None


def clear_profile_details_dialog() -> None:
    st.session_state.pop(PROFILE_DETAILS_DIALOG_KEY, None)
    st.session_state.pop(PROFILE_EDIT_MODE_KEY, None)


def request_profile_details_dialog(user_id: str) -> None:
    st.session_state[PROFILE_DETAILS_DIALOG_KEY] = {"user_id": user_id}


def get_profile_details_dialog_request() -> dict[str, str] | None:
    payload = st.session_state.get(PROFILE_DETAILS_DIALOG_KEY)
    return payload if isinstance(payload, dict) else None


def _format_user_label(profile) -> str:
    display_name = (profile.display_name or "").strip()
    if display_name:
        return f"{display_name} ({profile.user_id})"

    full_name = " ".join(part for part in [profile.first_name, profile.last_name] if part)
    if full_name.strip():
        return f"{full_name.strip()} ({profile.user_id})"

    return profile.user_id or "Unknown user"


def _switch_user(conversation_repository, user_id: str) -> None:
    latest = conversation_repository.get_latest_conversation(user_id)
    clear_feedback_state()
    clear_replay_state()
    clear_conversation_model_config_dialog()
    st.session_state.selected_user_id = user_id
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        conv = conversation_repository.create_conversation(
            user_id=user_id,
            metadata={"source": "streamlit"},
        )
        st.session_state.conversation_id = str(conv.id)
    st.query_params["uid"] = user_id
    st.query_params["cid"] = st.session_state.conversation_id
    st.session_state.loaded_cid = None
    st.session_state.messages = []
    st.session_state.debug_turns = []
    st.rerun()


def _delete_conversation(conversation_repository, conversation_id: str) -> None:
    selected_user_id = st.session_state.get("selected_user_id", "anonymous")
    conversation_repository.delete_conversation(UUID(conversation_id), user_id=selected_user_id)
    latest = conversation_repository.get_latest_conversation(selected_user_id)
    clear_feedback_state()
    clear_replay_state()
    clear_conversation_model_config_dialog()
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        conv = conversation_repository.create_conversation(
            user_id=selected_user_id,
            metadata={"source": "streamlit"},
        )
        st.session_state.conversation_id = str(conv.id)
    st.query_params["uid"] = selected_user_id
    st.query_params["cid"] = st.session_state.conversation_id
    st.session_state.loaded_cid = None
    st.session_state.messages = []
    st.rerun()


def _format_price(price: Decimal) -> str:
    normalized = format(price.normalize(), 'f')
    return f"${normalized} per 1M"


def _format_model_option_label(model_name: str, input_price: str, output_price: str) -> str:
    return f"{model_name} ({input_price} / {output_price})"


def build_model_config_rows(
    resolved_config: ConversationModelConfig,
    overrides: list,
) -> list[dict[str, str | None]]:
    override_map = {
        (entry.agent, entry.stage): entry.model
        for entry in overrides
    }
    rows: list[dict[str, str | None]] = []
    for spec in CONVERSATION_MODEL_CONFIG_SPECS:
        pricing = resolved_config.resolve_pricing(spec.agent, spec.stage)
        effective_model = resolved_config.resolve(spec.agent, spec.stage)
        input_price = _format_price(pricing.input_price_per_million_tokens)
        output_price = _format_price(pricing.output_price_per_million_tokens)
        option_to_model = {
            _format_model_option_label(model_name, _format_price(model_pricing.input_price_per_million_tokens), _format_price(model_pricing.output_price_per_million_tokens)): model_name
            for model_name, model_pricing in ConversationModelConfig.MODEL_PRICING_REGISTRY.items()
        }
        rows.append(
            {
                "agent": spec.agent,
                "stage": spec.stage,
                "label": spec.label,
                "effective_model": effective_model,
                "override_model": override_map.get((spec.agent, spec.stage)),
                "input_price": input_price,
                "output_price": output_price,
                "effective_model_option": _format_model_option_label(effective_model, input_price, output_price),
                "option_to_model": option_to_model,
            }
        )
    return rows


def _apply_model_config_form(conversation_repository, conversation_id: str, rows: list[dict[str, str | None]]) -> None:
    for row in rows:
        select_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}"
        default_option_label = f"Default ({row['effective_model_option']})"
        selected_value = st.session_state.get(select_key, default_option_label)
        if selected_value == default_option_label:
            conversation_repository.clear_conversation_model_config(
                UUID(conversation_id),
                row["agent"],
                row["stage"],
            )
        else:
            selected_model = row["option_to_model"].get(selected_value)
            if selected_model is None:
                raise KeyError(f"Unsupported model option label: {selected_value}")
            conversation_repository.upsert_conversation_model_config(
                UUID(conversation_id),
                row["agent"],
                row["stage"],
                selected_model,
            )


@st.dialog("Conversation Model Config", width="large")
def render_conversation_model_config_dialog(
    conversation_repository,
    conversation_id: str,
    title: str,
    replay_source_roundtrip_id: str | None = None,
) -> None:
    resolved_config = conversation_repository.resolve_conversation_model_config(UUID(conversation_id))
    overrides = conversation_repository.list_conversation_model_config(UUID(conversation_id))
    rows = build_model_config_rows(resolved_config, overrides)
    is_replay_mode = bool(replay_source_roundtrip_id)

    st.markdown(
        """<style>
        [data-testid="stDialog"] button[kind="primary"] {
            background-color: #2e7d32 !important;
            border-color: #2e7d32 !important;
            color: #ffffff !important;
        }
        [data-testid="stDialog"] button[kind="secondary"] {
            background-color: #b3261e !important;
            border-color: #b3261e !important;
            color: #ffffff !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    if is_replay_mode:
        st.caption(f"Review or adjust model overrides for `{title}` before replaying this conversation.")
    else:
        st.caption(f"Configure model overrides for `{title}`. Unset stages fall back to defaults.")

    current_section: str | None = None
    for row in rows:
        if row["agent"] != current_section:
            current_section = row["agent"]
            if rows.index(row) > 0:
                st.divider()
            st.subheader(MODEL_CONFIG_SECTION_TITLES.get(current_section, str(current_section)))

        select_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}"
        default_option_label = f"Default ({row['effective_model_option']})"
        options = [default_option_label, *row["option_to_model"].keys()]
        selected_value = row["effective_model_option"] if row["override_model"] else default_option_label
        selected_index = options.index(selected_value) if selected_value in options else 0

        col_select, col_info, col_reset = st.columns([3.8, 1.6, 1])
        with col_select:
            st.selectbox(
                row["label"],
                options,
                index=selected_index,
                key=select_key,
                help=f"Current effective model: {row['effective_model']}",
            )
        with col_info:
            st.caption(f"Effective: {row['effective_model']}")
            st.caption(f"Override: {row['override_model'] or 'default'}")
        with col_reset:
            st.write("")
            if st.button(
                "Reset",
                key=f"reset_model_config::{conversation_id}::{row['agent']}::{row['stage']}",
                type="secondary",
            ):
                conversation_repository.clear_conversation_model_config(
                    UUID(conversation_id),
                    row["agent"],
                    row["stage"],
                )
                st.rerun()

    if is_replay_mode:
        action_col, reset_col, cancel_col = st.columns(3)
        with action_col:
            if st.button("Accept replay", use_container_width=True, type="primary"):
                _apply_model_config_form(conversation_repository, conversation_id, rows)
                clear_conversation_model_config_dialog()
                st.session_state["pending_replay"] = {
                    "conversation_id": conversation_id,
                    "source_roundtrip_id": replay_source_roundtrip_id,
                }
                st.rerun()
        with reset_col:
            if st.button("Reset all", use_container_width=True, type="secondary"):
                conversation_repository.clear_all_conversation_model_config(UUID(conversation_id))
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                clear_conversation_model_config_dialog()
                st.rerun()
    else:
        save_col, reset_col, close_col = st.columns(3)
        with save_col:
            if st.button("Save config", use_container_width=True, type="primary"):
                _apply_model_config_form(conversation_repository, conversation_id, rows)
                clear_conversation_model_config_dialog()
                st.rerun()
        with reset_col:
            if st.button("Reset all", use_container_width=True, type="secondary"):
                conversation_repository.clear_all_conversation_model_config(UUID(conversation_id))
                st.rerun()
        with close_col:
            if st.button("Close", use_container_width=True):
                clear_conversation_model_config_dialog()
                st.rerun()


@st.dialog("Profile Details", width="large")
def render_profile_details_dialog(conversation_repository, user_id: str) -> None:
    user_profile_repository = get_user_profile_repo()
    selected_profile = user_profile_repository.get_profile(user_id)

    if selected_profile is None:
        st.error("User profile not found.")
        if st.button("Close", use_container_width=True):
            clear_profile_details_dialog()
            st.rerun()
        return

    conversation_count = conversation_repository.count_conversations(user_id)
    attribute_count = get_user_attribute_repo().count_attributes(user_id=user_id)
    full_name = " ".join(part for part in [selected_profile.first_name, selected_profile.last_name] if part).strip()
    edit_mode = bool(st.session_state.get(PROFILE_EDIT_MODE_KEY, False))

    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.user-profile-details) {
            background: linear-gradient(180deg, rgba(99, 136, 219, 0.12), rgba(99, 136, 219, 0.05));
            border-color: rgba(99, 136, 219, 0.28) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="user-profile-details"></div>', unsafe_allow_html=True)
        st.write(f"User ID: `{selected_profile.user_id}`")
        st.write(f"Display name: {selected_profile.display_name or '-'}")
        st.write(f"First name: {selected_profile.first_name or '-'}")
        st.write(f"Last name: {selected_profile.last_name or '-'}")
        st.write(f"Full name: {full_name or '-'}")
        st.write(f"Email: {selected_profile.email or '-'}")
        st.write(f"Conversations: {conversation_count}")
        st.write(f"Attributes: {attribute_count}")
        if selected_profile.created_at:
            st.caption(f"Created: {selected_profile.created_at}")
        if selected_profile.updated_at:
            st.caption(f"Updated: {selected_profile.updated_at}")

    if edit_mode:
        with st.form(f"profile_edit_form::{user_id}"):
            edited_first_name = st.text_input("First name", value=selected_profile.first_name or "")
            edited_last_name = st.text_input("Last name", value=selected_profile.last_name or "")
            edited_display_name = st.text_input("Display name", value=selected_profile.display_name or "")
            edited_email = st.text_input("Email", value=selected_profile.email or "")
            save_col, cancel_col = st.columns(2)
            with save_col:
                save_profile = st.form_submit_button("Save", type="primary", use_container_width=True)
            with cancel_col:
                cancel_edit = st.form_submit_button("Cancel", use_container_width=True)

        if save_profile:
            user_profile_repository.update_profile(
                user_id=user_id,
                first_name=edited_first_name.strip() or None,
                last_name=edited_last_name.strip() or None,
                display_name=edited_display_name.strip() or None,
                email=edited_email.strip() or None,
            )
            st.session_state[PROFILE_EDIT_MODE_KEY] = False
            st.rerun()
        if cancel_edit:
            st.session_state[PROFILE_EDIT_MODE_KEY] = False
            st.rerun()
    else:
        edit_col, close_col = st.columns(2)
        with edit_col:
            if st.button("Edit profile", use_container_width=True):
                st.session_state[PROFILE_EDIT_MODE_KEY] = True
                st.rerun()
        with close_col:
            if st.button("Close", use_container_width=True):
                clear_profile_details_dialog()
                st.rerun()


def render_sidebar(conversation_repository) -> None:
    st.title("LLM Agentic Chat")

    user_profile_repository = get_user_profile_repo()
    profiles = user_profile_repository.list_profiles(limit=100)
    if not profiles:
        user_profile_repository.ensure_profile("anonymous", display_name="Anonymous")
        profiles = user_profile_repository.list_profiles(limit=100)

    selected_user_id = st.session_state.get("selected_user_id") or profiles[0].user_id
    if selected_user_id not in {profile.user_id for profile in profiles}:
        selected_user_id = profiles[0].user_id
        st.session_state.selected_user_id = selected_user_id

    st.caption("User")
    user_col, create_col = st.columns([4, 1.2])
    with user_col:
        selected_user_id = st.selectbox(
            "User",
            options=[profile.user_id for profile in profiles],
            index=next((index for index, profile in enumerate(profiles) if profile.user_id == selected_user_id), 0),
            format_func=lambda user_id: _format_user_label(next(profile for profile in profiles if profile.user_id == user_id)),
            label_visibility="collapsed",
        )
    with create_col:
        if st.button("New user", use_container_width=True):
            st.session_state[USER_CREATE_FORM_KEY] = not st.session_state.get(USER_CREATE_FORM_KEY, False)

    previous_user_id = st.session_state.get("selected_user_id")
    if selected_user_id != previous_user_id:
        _switch_user(conversation_repository, selected_user_id)

    if st.button("View profile details", use_container_width=True):
        request_profile_details_dialog(user_id=selected_user_id)
        st.rerun()

    if st.session_state.get(USER_CREATE_FORM_KEY, False):
        with st.container(border=True):
            new_first_name = st.text_input("First name", key="new_user_first_name_input")
            new_last_name = st.text_input("Last name", key="new_user_last_name_input")
            new_display_name = st.text_input("Display name", key="new_user_display_name_input")
            create_user_col, cancel_user_col = st.columns(2)
            with create_user_col:
                if st.button("Create user", use_container_width=True, type="primary"):
                    resolved_user_id = str(uuid4())
                    user_profile_repository.ensure_profile(
                        resolved_user_id,
                        first_name=new_first_name.strip() or None,
                        last_name=new_last_name.strip() or None,
                        display_name=new_display_name.strip() or None,
                        metadata={"source": "streamlit"},
                    )
                    st.session_state[USER_CREATE_FORM_KEY] = False
                    st.session_state.pop("new_user_first_name_input", None)
                    st.session_state.pop("new_user_last_name_input", None)
                    st.session_state.pop("new_user_display_name_input", None)
                    _switch_user(conversation_repository, resolved_user_id)
            with cancel_user_col:
                if st.button("Cancel", use_container_width=True):
                    st.session_state[USER_CREATE_FORM_KEY] = False
                    st.rerun()

    current_id = st.session_state.conversation_id

    st.divider()
    conversations_label_col, new_chat_col = st.columns([2.8, 2.2])
    with conversations_label_col:
        st.caption("Conversations")
    with new_chat_col:
        if st.button(":material/add: New chat", use_container_width=True):
            clear_feedback_state()
            clear_replay_state()
            clear_conversation_model_config_dialog()
            conv = conversation_repository.create_conversation(user_id=selected_user_id, metadata={"source": "streamlit"})
            st.session_state.conversation_id = str(conv.id)
            st.query_params["uid"] = selected_user_id
            st.query_params["cid"] = st.session_state.conversation_id
            st.session_state.loaded_cid = None
            st.session_state.messages = []
            st.session_state.debug_turns = []
            st.rerun()

    conversations = conversation_repository.list_conversations(user_id=selected_user_id, limit=50)

    for c in conversations:
        cid = str(c.id)
        title = (c.title or "Untitled").strip()
        is_active = cid == current_id

        col_title, col_settings, col_delete = st.columns([6, 1.5, 1.5])
        with col_title:
            if is_active:
                st.markdown(
                    f'<div style="'
                    f'background:rgba(99,136,219,0.2);'
                    f'border-left:3px solid #6388db;'
                    f'border-radius:4px;'
                    f'padding:6px 10px;'
                    f'font-weight:600;'
                    f'white-space:nowrap;'
                    f'overflow:hidden;'
                    f'text-overflow:ellipsis;'
                    f'">{title}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(title, key=f"conv_{cid}", use_container_width=True):
                    clear_feedback_state()
                    clear_replay_state()
                    clear_conversation_model_config_dialog()
                    st.session_state.conversation_id = cid
                    st.query_params["uid"] = selected_user_id
                    st.query_params["cid"] = cid
                    st.session_state.loaded_cid = None
                    st.rerun()
        with col_settings:
            if st.button(
                ":material/settings:",
                key=f"cfg_{cid}",
                help="Conversation model config",
                use_container_width=True,
            ):
                request_conversation_model_config_dialog(conversation_id=cid, title=title)
                st.rerun()
        with col_delete:
            if st.button(
                ":material/delete:",
                key=f"del_{cid}",
                help="Delete conversation",
                use_container_width=True,
            ):
                _delete_conversation(conversation_repository, cid)

    dialog_request = get_conversation_model_config_dialog_request()
    if dialog_request:
        render_conversation_model_config_dialog(
            conversation_repository,
            dialog_request["conversation_id"],
            dialog_request["title"],
            dialog_request.get("replay_source_roundtrip_id"),
        )

    profile_dialog_request = get_profile_details_dialog_request()
    if profile_dialog_request:
        render_profile_details_dialog(
            conversation_repository,
            profile_dialog_request["user_id"],
        )
