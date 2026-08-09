from decimal import Decimal
from uuid import UUID

import streamlit as st

from common.model_constants import AVAILABLE_CHAT_MODELS
from conversation.models.conversation_model_config import CONVERSATION_MODEL_CONFIG_SPECS, ConversationModelConfig
from rendering.feedback import clear_feedback_state
from rendering.replay import clear_replay_state

MODEL_CONFIG_DIALOG_KEY = "conversation_model_config_dialog"


def clear_conversation_model_config_dialog() -> None:
    st.session_state.pop(MODEL_CONFIG_DIALOG_KEY, None)


def request_conversation_model_config_dialog(
    *,
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


def _delete_conversation(conversation_repository, conversation_id: str) -> None:
    conversation_repository.delete_conversation(UUID(conversation_id), user_id="anonymous")
    latest = conversation_repository.get_latest_conversation("anonymous")
    clear_feedback_state()
    clear_replay_state()
    clear_conversation_model_config_dialog()
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        conv = conversation_repository.create_conversation(
            user_id="anonymous",
            metadata={"source": "streamlit"},
        )
        st.session_state.conversation_id = str(conv.id)
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

    for row in rows:
        select_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}"
        default_option_label = f"Default ({row['effective_model_option']})"
        model_options = [row["option_to_model"][model_option] for model_option in row["option_to_model"]]
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


def render_sidebar(conversation_repository) -> None:
    st.title("LLM Agentic Chat")

    if st.button(":material/add: New chat"):
        clear_feedback_state()
        clear_replay_state()
        clear_conversation_model_config_dialog()
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id
        st.session_state.loaded_cid = None
        st.session_state.messages = []
        st.session_state.debug_turns = []
        st.rerun()

    current_id = st.session_state.conversation_id

    st.divider()
    st.caption("Conversations")
    conversations = conversation_repository.list_conversations(user_id="anonymous", limit=50)

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
