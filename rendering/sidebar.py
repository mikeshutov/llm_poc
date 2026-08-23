from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import streamlit as st

from llm.repository.repo_factory import get_conversation_model_config_repo
from llm.conversation_model_config import CONVERSATION_MODEL_CONFIG_SPECS, ConversationModelConfig, EVALUATOR_STAGE, PLANNER_STAGE
from conversation.models.replay_models import PreparedReplayConversation
from personalization.profile.repository.repo_factory import get_user_profile_repo
from personalization.user_attributes.repository.repo_factory import get_user_attribute_repo
from request_orchestrator.agent_runner.models.agent_profile import AgentExecutionStrategy
from request_orchestrator.agents.models.user_agent import UserAgentModelConfig
from request_orchestrator.agents.repository.repo_factory import get_user_agent_repo
from rendering.feedback import clear_feedback_state
from rendering.replay import clear_replay_state
from rendering.sources import clear_sources_panel
from tool.tools import TOOL_CATEGORIES

MODEL_CONFIG_DIALOG_KEY = "conversation_model_config_dialog"
PENDING_REPLAY_PREPARE_KEY = "pending_replay_prepare"
MODEL_CONFIG_SECTION_TITLES = {
    "main_agent": "Main Agent",
    "profile_agent": "Profile Management Agent",
    "shared": "Shared",
}
USER_CREATE_FORM_KEY = "show_create_user_form"
PROFILE_DETAILS_DIALOG_KEY = "profile_details_dialog"
PROFILE_EDIT_MODE_KEY = "profile_details_edit_mode"
USER_AGENTS_DIALOG_KEY = "user_agents_dialog"
USER_AGENTS_CREATE_MODE_KEY = "user_agents_create_mode"
USER_AGENTS_EDIT_AGENT_ID_KEY = "user_agents_edit_agent_id"
USER_AGENT_STAGE_TITLES = {
    PLANNER_STAGE: "Planner",
    EVALUATOR_STAGE: "Evaluator",
}


def clear_conversation_model_config_dialog() -> None:
    st.session_state.pop(MODEL_CONFIG_DIALOG_KEY, None)


def request_conversation_model_config_dialog(
    conversation_id: str,
    title: str,
    replay_source_roundtrip_id: str | None = None,
    replay_context: PreparedReplayConversation | None = None,
) -> None:
    st.session_state[MODEL_CONFIG_DIALOG_KEY] = {
        "conversation_id": conversation_id,
        "title": title,
        "replay_source_roundtrip_id": replay_source_roundtrip_id,
        "replay_context": None if replay_context is None else replay_context.model_dump(),
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


def clear_user_agents_dialog() -> None:
    st.session_state.pop(USER_AGENTS_DIALOG_KEY, None)
    st.session_state.pop(USER_AGENTS_CREATE_MODE_KEY, None)
    st.session_state.pop(USER_AGENTS_EDIT_AGENT_ID_KEY, None)


def request_user_agents_dialog(user_id: str) -> None:
    st.session_state[USER_AGENTS_DIALOG_KEY] = {"user_id": user_id}


def get_user_agents_dialog_request() -> dict[str, str] | None:
    payload = st.session_state.get(USER_AGENTS_DIALOG_KEY)
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
    clear_sources_panel()
    clear_conversation_model_config_dialog()
    st.session_state.selected_user_id = user_id
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        st.session_state.conversation_id = str(
            conversation_repository.create_conversation(
                user_id=user_id,
                metadata={"source": "streamlit"},
            ).id
        )
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
    clear_sources_panel()
    clear_conversation_model_config_dialog()
    if latest:
        st.session_state.conversation_id = str(latest.id)
    else:
        st.session_state.conversation_id = str(
            conversation_repository.create_conversation(
                user_id=selected_user_id,
                metadata={"source": "streamlit"},
            ).id
        )
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


def _build_model_option_lookup() -> dict[str, dict[str, str]]:
    grouped_options: dict[str, dict[str, str]] = {}
    for provider, model_names in ConversationModelConfig.model_names_by_provider().items():
        grouped_options[provider] = {
            _format_model_option_label(
                model_name,
                _format_price(ConversationModelConfig.resolve_model_pricing(provider, model_name).input_price_per_million_tokens),
                _format_price(ConversationModelConfig.resolve_model_pricing(provider, model_name).output_price_per_million_tokens),
            ): model_name
            for model_name in model_names
        }
    return grouped_options


def _default_user_agent_provider() -> str:
    return ConversationModelConfig.build_default().main_agent.planner.provider


def _default_user_agent_model_selection(stage: str) -> tuple[str, str]:
    default_config = ConversationModelConfig.build_default()
    if stage == PLANNER_STAGE:
        selection = default_config.main_agent.planner
        return selection.provider, selection.model
    if stage == EVALUATOR_STAGE:
        selection = default_config.shared.evaluator
        return selection.provider, selection.model
    return _default_user_agent_provider(), ""


def _build_user_agent_model_config_inputs(
    *,
    widget_key_prefix: str,
    execution_strategy: AgentExecutionStrategy,
) -> list[UserAgentModelConfig]:
    provider_options = ConversationModelConfig.model_names_by_provider()
    model_configs: list[UserAgentModelConfig] = []
    for stage in execution_strategy.required_model_stages():
        provider_key = f"{widget_key_prefix}::{stage}::provider"
        model_key = f"{widget_key_prefix}::{stage}::model"
        provider_choices = list(provider_options.keys())
        default_provider, default_model = _default_user_agent_model_selection(stage)
        if provider_key not in st.session_state or st.session_state[provider_key] not in provider_choices:
            st.session_state[provider_key] = (
                default_provider
                if default_provider in provider_choices
                else _default_user_agent_provider()
            )
        provider_col, model_col = st.columns(2)
        with provider_col:
            selected_provider = st.selectbox(
                f"{USER_AGENT_STAGE_TITLES.get(stage, stage)} provider",
                provider_choices,
                key=provider_key,
                format_func=ConversationModelConfig.provider_display_name,
            )
        model_choices = provider_options.get(selected_provider, [])
        if model_key not in st.session_state or st.session_state[model_key] not in model_choices:
            st.session_state[model_key] = (
                default_model
                if selected_provider == default_provider and default_model in model_choices
                else (model_choices[0] if model_choices else "")
            )
        with model_col:
            selected_model = st.selectbox(
                f"{USER_AGENT_STAGE_TITLES.get(stage, stage)} model",
                model_choices,
                key=model_key,
            )
        model_configs.append(
            UserAgentModelConfig(
                stage=stage,
                provider=selected_provider,
                model=selected_model,
            )
        )
    return model_configs


def _initialize_user_agent_edit_form(user_id: str, user_agent) -> None:
    widget_prefix = f"user_agent_edit::{user_id}::{user_agent.id}"
    st.session_state[f"{widget_prefix}::name"] = user_agent.name
    st.session_state[f"{widget_prefix}::description"] = user_agent.description
    st.session_state[f"{widget_prefix}::execution_strategy"] = user_agent.execution_strategy.value
    st.session_state[f"{widget_prefix}::allowed_categories"] = list(user_agent.allowed_categories)
    st.session_state[f"{widget_prefix}::planner_instruction"] = user_agent.planner_instruction
    st.session_state[f"{widget_prefix}::planner_rules"] = user_agent.planner_rules
    st.session_state[f"{widget_prefix}::max_turns"] = user_agent.max_turns
    for model_config in user_agent.model_configs:
        st.session_state[f"{widget_prefix}::{model_config.stage}::provider"] = model_config.provider
        st.session_state[f"{widget_prefix}::{model_config.stage}::model"] = model_config.model


def _resolve_config_model(config: ConversationModelConfig, agent: str, stage: str) -> str:
    resolved = config.resolve(agent, stage)
    if isinstance(resolved, str):
        return resolved
    return getattr(resolved, "model", str(resolved))


def _resolve_config_provider(config: ConversationModelConfig, agent: str, stage: str) -> str:
    resolver = getattr(config, "resolve_provider", None)
    if callable(resolver):
        return resolver(agent, stage)

    selection_resolver = getattr(config, "resolve_selection", None)
    if callable(selection_resolver):
        selection = selection_resolver(agent, stage)
        provider = getattr(selection, "provider", "")
        if isinstance(provider, str) and provider.strip():
            return provider

    return "openai"


def build_model_config_rows(
    resolved_config: ConversationModelConfig,
    overrides: list,
) -> list[dict[str, str | None]]:
    override_map = {
        (entry.agent, entry.stage): entry.model
        for entry in overrides
    }
    provider_options = _build_model_option_lookup()
    option_to_model = {
        option_label: model_name
        for provider_option_map in provider_options.values()
        for option_label, model_name in provider_option_map.items()
    }
    rows: list[dict[str, str | None]] = []
    for spec in CONVERSATION_MODEL_CONFIG_SPECS:
        pricing = resolved_config.resolve_pricing(spec.agent, spec.stage)
        effective_model = _resolve_config_model(resolved_config, spec.agent, spec.stage)
        effective_provider_key = _resolve_config_provider(resolved_config, spec.agent, spec.stage)
        input_price = _format_price(pricing.input_price_per_million_tokens)
        output_price = _format_price(pricing.output_price_per_million_tokens)
        rows.append(
            {
                "agent": spec.agent,
                "stage": spec.stage,
                "label": spec.label,
                "effective_model": effective_model,
                "effective_provider_key": effective_provider_key,
                "override_model": override_map.get((spec.agent, spec.stage)),
                "override_provider": next(
                    (
                        entry.provider
                        for entry in overrides
                        if entry.agent == spec.agent and entry.stage == spec.stage
                    ),
                    None,
                ),
                "input_price": input_price,
                "output_price": output_price,
                "effective_provider": ConversationModelConfig.provider_display_name(effective_provider_key),
                "effective_model_option": _format_model_option_label(effective_model, input_price, output_price),
                "provider_options": provider_options,
                "option_to_model": option_to_model,
            }
        )
    return rows


def _apply_model_config_form(model_config_repository, conversation_id: str, rows: list[dict[str, str | None]]) -> None:
    for row in rows:
        select_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}::model"
        default_option_label = f"Default ({row['effective_model_option']})"
        selected_value = st.session_state.get(select_key, default_option_label)
        if selected_value == default_option_label:
            model_config_repository.clear(
                UUID(conversation_id),
                row["agent"],
                row["stage"],
            )
        else:
            selected_model = row["option_to_model"].get(selected_value)
            if selected_model is None:
                raise KeyError(f"Unsupported model option label: {selected_value}")
            selected_provider = st.session_state.get(
                f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}::provider",
                row["effective_provider_key"],
            )
            model_config_repository.upsert(
                UUID(conversation_id),
                row["agent"],
                row["stage"],
                selected_provider,
                selected_model,
            )


def _render_model_config_section(model_config_repository, conversation_id: str, rows: list[dict[str, str | None]]) -> None:
    for row in rows:
        provider_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}::provider"
        model_key = f"conversation_model_config::{conversation_id}::{row['agent']}::{row['stage']}::model"
        default_option_label = f"Default ({row['effective_model_option']})"
        provider_options = row["provider_options"]
        providers = list(provider_options.keys())

        selected_provider = st.session_state.get(provider_key, row["effective_provider_key"])
        if selected_provider not in providers:
            selected_provider = row["effective_provider_key"] if row["effective_provider_key"] in providers else providers[0]

        available_model_options = list(provider_options.get(selected_provider, {}).keys())
        if selected_provider == row["effective_provider_key"]:
            available_model_options = [default_option_label, *available_model_options]

        selected_value = st.session_state.get(model_key)
        if selected_value not in available_model_options:
            if selected_provider == row["effective_provider_key"]:
                selected_value = row["effective_model_option"] if row["override_model"] else default_option_label
                if selected_value not in available_model_options:
                    selected_value = default_option_label
            else:
                selected_value = available_model_options[0] if available_model_options else ""
            st.session_state[model_key] = selected_value
        if provider_key not in st.session_state:
            st.session_state[provider_key] = selected_provider
        if model_key not in st.session_state:
            st.session_state[model_key] = selected_value

        st.markdown(f"**{row['label']}**")
        col_provider, col_model, col_reset = st.columns([1.4, 4.6, 1])
        with col_provider:
            st.selectbox(
                "Provider",
                providers,
                key=provider_key,
                format_func=ConversationModelConfig.provider_display_name,
                label_visibility="collapsed",
                help="Filter model options by provider.",
            )
        with col_model:
            st.selectbox(
                "Model",
                available_model_options,
                key=model_key,
                label_visibility="collapsed",
                help=f"Current effective model: {row['effective_model']}",
            )
        with col_reset:
            st.write("")
            if st.button(
                "Reset",
                key=f"reset_model_config::{conversation_id}::{row['agent']}::{row['stage']}",
                type="secondary",
            ):
                model_config_repository.clear(
                    UUID(conversation_id),
                    row["agent"],
                    row["stage"],
                )
                st.rerun()


@st.dialog("Conversation Model Config", width="large")
def render_conversation_model_config_dialog(
    conversation_repository,
    conversation_id: str,
    title: str,
    replay_source_roundtrip_id: str | None = None,
    replay_context: PreparedReplayConversation | None = None,
) -> None:
    model_config_repository = get_conversation_model_config_repo()
    resolved_config = model_config_repository.resolve(UUID(conversation_id))
    overrides = model_config_repository.list(UUID(conversation_id))
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

    rows_by_agent: dict[str, list[dict[str, str | None]]] = {}
    for row in rows:
        rows_by_agent.setdefault(str(row["agent"]), []).append(row)

    section_agents = list(rows_by_agent.keys())
    section_tabs = st.tabs([MODEL_CONFIG_SECTION_TITLES.get(agent, agent) for agent in section_agents])
    for tab, agent in zip(section_tabs, section_agents):
        with tab:
            _render_model_config_section(
                model_config_repository,
                conversation_id,
                rows_by_agent[agent],
            )

    if is_replay_mode:
        action_col, reset_col, cancel_col = st.columns(3)
        with action_col:
            if st.button("Accept replay", use_container_width=True, type="primary"):
                _apply_model_config_form(model_config_repository, conversation_id, rows)
                clear_conversation_model_config_dialog()
                if replay_context is None:
                    raise ValueError("replay_context is required in replay mode")
                st.session_state[PENDING_REPLAY_PREPARE_KEY] = replay_context.model_dump()
                st.rerun()
        with reset_col:
            if st.button("Reset all", use_container_width=True, type="secondary"):
                model_config_repository.clear_all(UUID(conversation_id))
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                clear_conversation_model_config_dialog()
                st.rerun()
    else:
        save_col, reset_col, close_col = st.columns(3)
        with save_col:
            if st.button("Save config", use_container_width=True, type="primary"):
                _apply_model_config_form(model_config_repository, conversation_id, rows)
                clear_conversation_model_config_dialog()
                st.rerun()
        with reset_col:
            if st.button("Reset all", use_container_width=True, type="secondary"):
                model_config_repository.clear_all(UUID(conversation_id))
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
        identity_tab, tone_tab, activity_tab = st.tabs(["Identity", "Tone", "Activity"])

        with identity_tab:
            st.write(f"**User ID:** `{selected_profile.user_id}`")
            st.write(f"**Display name:** {selected_profile.display_name or '-'}")
            st.write(f"**First name:** {selected_profile.first_name or '-'}")
            st.write(f"**Last name:** {selected_profile.last_name or '-'}")
            st.write(f"**Full name:** {full_name or '-'}")
            st.write(f"**Email:** {selected_profile.email or '-'}")
            if selected_profile.created_at:
                st.caption(f"**Created:** {selected_profile.created_at}")
            if selected_profile.updated_at:
                st.caption(f"**Updated:** {selected_profile.updated_at}")

        with tone_tab:
            if selected_profile.tone is None:
                st.write("-")
            else:
                st.write(f"**Verbosity:** {selected_profile.tone.verbosity or '-'}")
                st.write(f"**Formality:** {selected_profile.tone.formality or '-'}")
                st.write(f"**Directness:** {selected_profile.tone.directness or '-'}")
                st.write(f"**Humor:** {selected_profile.tone.humor or '-'}")
                st.write(f"**Technical depth:** {selected_profile.tone.technical_depth or '-'}")

        with activity_tab:
            st.write(f"**Conversations:** {conversation_count}")
            st.write(f"**Attributes:** {attribute_count}")

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


@st.dialog("User Agents", width="large")
def render_user_agents_dialog(user_id: str) -> None:
    user_agent_repository = get_user_agent_repo()
    user_agents = user_agent_repository.list_for_user(user_id, is_active=None)
    create_mode = bool(st.session_state.get(USER_AGENTS_CREATE_MODE_KEY, False))
    edit_agent_id = str(st.session_state.get(USER_AGENTS_EDIT_AGENT_ID_KEY, "")).strip()
    edited_agent = next((agent for agent in user_agents if str(agent.id) == edit_agent_id), None)
    if edit_agent_id and edited_agent is None:
        st.session_state.pop(USER_AGENTS_EDIT_AGENT_ID_KEY, None)
    form_agent = edited_agent
    form_visible = create_mode or form_agent is not None

    header_col, action_col = st.columns([4.5, 1.5], vertical_alignment="center")
    with header_col:
        st.caption(f"Manage planner-only user agents for `{user_id}`.")
    with action_col:
        if st.button(
            "Create agent" if not form_visible else "Hide form",
            key=f"user_agents_toggle_create::{user_id}",
            use_container_width=True,
            type="primary" if not form_visible else "secondary",
        ):
            st.session_state[USER_AGENTS_CREATE_MODE_KEY] = not form_visible
            st.session_state.pop(USER_AGENTS_EDIT_AGENT_ID_KEY, None)
            st.rerun()

    if form_visible:
        is_editing = form_agent is not None
        widget_prefix = (
            f"user_agent_edit::{user_id}::{form_agent.id}"
            if is_editing
            else f"user_agent_create::{user_id}"
        )
        agent_name = st.text_input(
            "Agent name",
            key=f"{widget_prefix}::name",
            disabled=is_editing,
            help="Agent names cannot be changed after creation.",
        )
        agent_description = st.text_area(
            "Description",
            placeholder="What this agent is for.",
            key=f"{widget_prefix}::description",
        )
        execution_strategy = AgentExecutionStrategy(
            st.selectbox(
                "Execution strategy",
                options=[strategy.value for strategy in AgentExecutionStrategy],
                key=f"{widget_prefix}::execution_strategy",
            )
        )
        allowed_category_names = st.multiselect(
            "Tool categories",
            options=sorted(TOOL_CATEGORIES.keys()),
            help="These categories determine which tools the agent can use.",
            key=f"{widget_prefix}::allowed_categories",
        )
        planner_instruction = st.text_area(
            "Planner instruction",
            placeholder="Core planner behavior for this user agent.",
            height=140,
            key=f"{widget_prefix}::planner_instruction",
        )
        planner_rules = st.text_area(
            "Planner rules",
            placeholder="Optional extra rules or constraints.",
            height=120,
            key=f"{widget_prefix}::planner_rules",
        )
        max_turns = st.number_input(
            "Max turns",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key=f"{widget_prefix}::max_turns",
        )
        st.caption("Model config")
        model_configs = _build_user_agent_model_config_inputs(
            widget_key_prefix=widget_prefix,
            execution_strategy=execution_strategy,
        )
        save_col, cancel_col = st.columns(2)
        with save_col:
            save_agent = st.button(
                "Save changes" if is_editing else "Create agent",
                type="primary",
                use_container_width=True,
            )
        with cancel_col:
            cancel_form = st.button("Cancel", use_container_width=True)

        if save_agent:
            user_agent_repository.upsert(
                user_id=user_id,
                name=agent_name,
                description=agent_description.strip(),
                execution_strategy=execution_strategy,
                allowed_categories=allowed_category_names,
                planner_instruction=planner_instruction,
                planner_rules=planner_rules.strip(),
                max_turns=int(max_turns),
                is_active=form_agent.is_active if form_agent is not None else True,
                model_configs=model_configs,
                metadata=(
                    dict(form_agent.metadata)
                    if form_agent is not None
                    else {"source": "streamlit"}
                ),
            )
            st.session_state[USER_AGENTS_CREATE_MODE_KEY] = False
            st.session_state.pop(USER_AGENTS_EDIT_AGENT_ID_KEY, None)
            st.rerun()
        if cancel_form:
            st.session_state[USER_AGENTS_CREATE_MODE_KEY] = False
            st.session_state.pop(USER_AGENTS_EDIT_AGENT_ID_KEY, None)
            st.rerun()

    if not user_agents:
        st.info("No user agents yet.")
    else:
        for user_agent in user_agents:
            with st.container(border=True):
                title_col, status_col, edit_col, action_col = st.columns([3.5, 1.2, 1.3, 1.5], vertical_alignment="center")
                with title_col:
                    st.markdown(f"**{user_agent.name}**")
                with status_col:
                    st.caption("Active" if user_agent.is_active else "Disabled")
                with edit_col:
                    if st.button(
                        "Edit",
                        key=f"edit_user_agent::{user_id}::{user_agent.id}",
                        use_container_width=True,
                    ):
                        _initialize_user_agent_edit_form(user_id, user_agent)
                        st.session_state[USER_AGENTS_CREATE_MODE_KEY] = False
                        st.session_state[USER_AGENTS_EDIT_AGENT_ID_KEY] = str(user_agent.id)
                        st.rerun()
                with action_col:
                    if user_agent.is_active:
                        if st.button(
                            "Disable",
                            key=f"disable_user_agent::{user_id}::{user_agent.name}",
                            use_container_width=True,
                            type="secondary",
                        ):
                            user_agent_repository.set_active(
                                user_id,
                                user_agent.name,
                                is_active=False,
                            )
                            st.rerun()
                    else:
                        if st.button(
                            "Enable",
                            key=f"enable_user_agent::{user_id}::{user_agent.name}",
                            use_container_width=True,
                            type="primary",
                        ):
                            user_agent_repository.set_active(
                                user_id,
                                user_agent.name,
                                is_active=True,
                            )
                            st.rerun()

                if user_agent.description.strip():
                    st.write(user_agent.description)
                st.caption(f"Strategy: {user_agent.execution_strategy.value}")
                st.caption(
                    f"Categories: {', '.join(user_agent.allowed_categories) if user_agent.allowed_categories else '-'}"
                )
                if user_agent.model_configs:
                    st.caption(
                        "Models: "
                        + ", ".join(
                            f"{USER_AGENT_STAGE_TITLES.get(config.stage, config.stage)}={ConversationModelConfig.provider_display_name(config.provider)} / {config.model}"
                            for config in user_agent.model_configs
                        )
                    )
                st.caption(f"Max turns: {user_agent.max_turns}")

    if st.button("Close", key=f"close_user_agents_dialog::{user_id}", use_container_width=True):
        clear_user_agents_dialog()
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

    if st.button("View User Agents", use_container_width=True):
        request_user_agents_dialog(user_id=selected_user_id)
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
            clear_sources_panel()
            clear_conversation_model_config_dialog()
            st.session_state.conversation_id = str(
                conversation_repository.create_conversation(
                    user_id=selected_user_id,
                    metadata={"source": "streamlit"},
                ).id
            )
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
                    clear_sources_panel()
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
        replay_context = dialog_request.get("replay_context")
        render_conversation_model_config_dialog(
            conversation_repository,
            dialog_request["conversation_id"],
            dialog_request["title"],
            dialog_request.get("replay_source_roundtrip_id"),
            None if replay_context is None else PreparedReplayConversation.model_validate(replay_context),
        )

    profile_dialog_request = get_profile_details_dialog_request()
    if profile_dialog_request:
        render_profile_details_dialog(
            conversation_repository,
            profile_dialog_request["user_id"],
        )

    user_agents_dialog_request = get_user_agents_dialog_request()
    if user_agents_dialog_request:
        render_user_agents_dialog(user_agents_dialog_request["user_id"])
