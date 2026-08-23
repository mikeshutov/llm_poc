from __future__ import annotations

from langsmith import traceable

from llm.clients.embeddings import embed_text
from request_orchestrator.agents.repository.repo_factory import get_user_agent_repo
from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.agents.agent_selection_query import build_agent_selection_query


@traceable(name="Load User Agents Node")
def load_user_agents(main_state: MainState) -> MainState:
    user_id = (main_state.execution_context.user_profile.user_id or "").strip()
    if not user_id:
        return main_state

    task = main_state.task.strip()
    if not task:
        return main_state

    user_agent_repository = get_user_agent_repo()
    if not user_agent_repository.list_for_user(user_id):
        return main_state

    user_agents = user_agent_repository.list_relevant_for_user(
        user_id,
        query_embedding=embed_text(
            build_agent_selection_query(
                current_user_request=task,
                conversation_context=main_state.execution_context.conversation_context,
            )
        ),
    )
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
