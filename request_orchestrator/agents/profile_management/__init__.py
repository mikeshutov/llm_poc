__all__ = ["PROFILE_MANAGEMENT_PROFILE", "run_agent"]


def __getattr__(name: str):
    if name == "run_agent":
        from request_orchestrator.agents.profile_management.agent import run_agent

        return run_agent
    if name == "PROFILE_MANAGEMENT_PROFILE":
        from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE

        return PROFILE_MANAGEMENT_PROFILE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
