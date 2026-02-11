"""
Interaction registry module for managing interaction class registration and initialization.

Provides:
- InteractionRegistry: Class-level registry with decorator-based registration
- initialize_interactions_from_args: Args-driven interaction initialization
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InteractionRegistry:
    """Registry for interaction classes.
    
    Usage:
        @InteractionRegistry.register("code")
        class CodeInteraction(BaseInteraction):
            ...
        
        interaction_map = initialize_interactions_from_args(args)
    """
    _registry: dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an interaction class.
        
        Args:
            name: Name to register the interaction under
        """
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_interaction_class(cls, name: str):
        """Get a registered interaction class by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown interaction: {name}. Available: {available}")
        return cls._registry[name]
    
    @classmethod
    def list_interactions(cls) -> list[str]:
        """List all registered interaction names."""
        return list(cls._registry.keys())


def initialize_interactions_from_args(args) -> dict[str, 'BaseInteraction']:
    """Initialize interactions based on args.
    
    Args:
        args: Namespace with:
            - interaction_name: str or None, interaction to initialize (default: None = no interaction)
            - sandbox_url, sandbox_default_time_limit_s, etc. for config
    
    Returns:
        Dict mapping interaction name to BaseInteraction instance (empty if None)
    """
    interaction_name = getattr(args, "interaction_name", None)
    
    if not interaction_name:
        return {}
    
    # Import here to trigger registration via decorators
    from . import interactions  # noqa: F401
    
    interaction_cls = InteractionRegistry.get_interaction_class(interaction_name)
    
    config = {
        "tool_config": {
            "sandbox_url": getattr(args, "sandbox_url", None),
            "timeout": getattr(args, "sandbox_default_time_limit_s", 10),
            "memory_limit": getattr(args, "sandbox_default_memory_limit_mb", 1024),
            "execution_num_workers": getattr(args, "execution_num_workers", 32),
        },
        "use_local_sandbox": getattr(args, "use_local_sandbox", False),
    }
    
    interaction = interaction_cls(config)
    logger.info(f"Initialized interaction '{interaction_name}' with class '{interaction_cls.__name__}'")
    
    return {interaction_name: interaction}
