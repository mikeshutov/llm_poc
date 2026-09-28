__all__ = [
    "attribute_discovery",
    "capability_discovery",
]


def __getattr__(name: str):
    if name == "attribute_discovery":
        from request_orchestrator.shared.discovery.attribute_discovery import attribute_discovery
        return attribute_discovery
    if name == "capability_discovery":
        from request_orchestrator.shared.discovery.capability_discovery import capability_discovery
        return capability_discovery
    raise AttributeError(name)
