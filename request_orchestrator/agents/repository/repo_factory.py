from request_orchestrator.agents.repository.user_agent_repository import UserAgentRepository


def get_user_agent_repo() -> UserAgentRepository:
    return UserAgentRepository()
