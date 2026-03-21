"""Agent registry for modular agent discovery and instantiation."""


from src.agents.base import BaseAgent
from src.logging_config.structured import get_logger

_REGISTRY: dict[str, type[BaseAgent]] = {}
_logger = get_logger(__name__)


def register_agent(name: str):
    """Decorator to register an agent class in the global registry.

    Usage:
        @register_agent("data_profiler")
        class DataProfilerAgent(ReActAgent):
            ...
    """
    def decorator(cls: type[BaseAgent]) -> type[BaseAgent]:
        _REGISTRY[name] = cls
        if not getattr(cls, "WRITES_STATE_FIELDS", None):
            _logger.debug("agent_no_writes_declared", agent=name)
        return cls
    return decorator


def get_agent_registry() -> dict[str, type[BaseAgent]]:
    """Return a copy of the current agent registry.

    Unlike create_all_agents() which instantiates, this returns
    the class registry for introspection without side effects.
    """
    return dict(_REGISTRY)


def create_all_agents() -> dict[str, BaseAgent]:
    """Instantiate all registered agents and return them keyed by name."""
    return {name: cls() for name, cls in _REGISTRY.items()}
