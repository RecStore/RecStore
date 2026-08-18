"""OptimizationPluginRegistry — singleton registry for optimization strategies.

Usage::

    from recstore.optim import OptimizationPluginRegistry

    plugin = OptimizationPluginRegistry.create(
        "bagpipe",
        embedding_module=...,
        kv_client=...,
        lookahead=4,
    )

    print(OptimizationPluginRegistry.available())  # ["none", "bagpipe", ...]
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .plugin import NullOptimizationPlugin, OptimizationPlugin


class OptimizationPluginRegistry:
    """Singleton registry — look up optimization strategy factories by name."""

    _instance: Optional["OptimizationPluginRegistry"] = None
    _factories: Dict[str, Callable[..., OptimizationPlugin]] = {}

    @classmethod
    def instance(cls) -> "OptimizationPluginRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(
        cls, name: str, factory: Callable[..., OptimizationPlugin]
    ) -> None:
        """Register a plugin factory under *name*.

        ``factory`` should be a callable (typically a classmethod ``create``)
        that accepts keyword arguments and returns an :class:`OptimizationPlugin`.
        """
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> OptimizationPlugin:
        """Create a plugin instance by registered name.

        ``"none"`` or ``None`` returns :class:`NullOptimizationPlugin`.
        """
        if name == "none" or name is None:
            return NullOptimizationPlugin()
        factory = cls._factories.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown optimization plugin: {name!r}. "
                f"Available: {cls.available()}"
            )
        return factory(**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Return all registered plugin names, including the built-in ``"none"``."""
        return ["none"] + sorted(cls._factories.keys())
