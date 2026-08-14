__all__ = ["run_agent"]


def __getattr__(name: str):
    if name == "run_agent":
        from request_orchestrator.agents.main_agent.agent import run_agent

        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
