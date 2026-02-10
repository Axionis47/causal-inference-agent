"""Agent registry for modular agent discovery and instantiation."""


from src.agents.base import BaseAgent

_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(name: str):
    """Decorator to register an agent class in the global registry.

    Usage:
        @register_agent("data_profiler")
        class DataProfilerAgent(ReActAgent):
            ...
    """
    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_agent_registry() -> dict[str, type[BaseAgent]]:
    """Return a copy of the current agent registry."""
    return dict(_REGISTRY)


def create_all_agents() -> dict[str, BaseAgent]:
    """Instantiate all registered agents and return them keyed by name."""
    return {name: cls() for name, cls in _REGISTRY.items()}
