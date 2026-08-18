from __future__ import annotations

from langsmith import traceable

from request_orchestrator.agents.repository.repo_factory import get_user_agent_repo
from request_orchestrator.models.main_state import MainState


@traceable(name="Load User Agents Node")
def load_user_agents(main_state: MainState) -> MainState:
    user_id = (main_state.execution_context.user_profile.user_id or "").strip()
    if not user_id:
        return main_state

    user_agents = get_user_agent_repo().list_for_user(user_id)
    loaded_profiles = [
        user_agent.to_agent_profile()
        for user_agent in user_agents
    ]
    if not loaded_profiles:
        return main_state

    existing_names = {profile.name for profile in main_state.agent_profiles}
    for profile in loaded_profiles:
        if profile.name in existing_names:
            continue
        main_state.agent_profiles.append(profile)
        existing_names.add(profile.name)
    main_state.initialize_agent_states()
    return main_state
