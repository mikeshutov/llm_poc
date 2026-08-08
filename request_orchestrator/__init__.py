def __getattr__(name: str):
    if name == "run_request_orchestrator_for_query":
        from request_orchestrator.service import run_request_orchestrator_for_query

        return run_request_orchestrator_for_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_request_orchestrator_for_query"]
