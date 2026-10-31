from request_orchestrator.agents.repository.user_agent_repository import AgentRepository


def get_agent_repo() -> AgentRepository:
    return AgentRepository()
