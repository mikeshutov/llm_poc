__all__ = ["AgentProfile"]


def __getattr__(name: str):
    if name == "AgentProfile":
        from request_orchestrator.agents.models.agent_profile import AgentProfile

        return AgentProfile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
