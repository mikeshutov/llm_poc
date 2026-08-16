def __getattr__(name: str):
    if name in {"AgentRunner", "AgentStratagy"}:
        from request_orchestrator.agent_runner.runner import AgentRunner, AgentStratagy

        return {
            "AgentRunner": AgentRunner,
            "AgentStratagy": AgentStratagy,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentRunner", "AgentStratagy"]
