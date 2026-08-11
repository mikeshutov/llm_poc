from importlib import import_module

__all__ = ["run_agent"]


def __getattr__(name: str):
    if name != "run_agent":
        raise AttributeError(name)
    module = import_module("request_orchestrator.agents.main_agent")
    value = getattr(module, name)
    globals()[name] = value
    return value
