"""
Chat template formatter module for handling model-specific message encoding.

This module provides:
- ChatTemplateFormatter: Abstract base class for chat template formatting
- DefaultChatFormatter: Default implementation using tokenizer.apply_chat_template

Design aligned with existing tool_parser.py patterns and ToolParser registry.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from examples.code.global_utils import get_event_loop

logger = logging.getLogger(__name__)


class ChatTemplateFormatter(ABC):
    """
    Abstract base class for chat template formatting.
    
    Responsibilities:
    - Convert messages to token ids (encode)
    - Handle model-specific special tokens and formats
    
    Usage:
        @ChatTemplateFormatter.register("my_formatter")
        class MyFormatter(ChatTemplateFormatter):
            ...
        
        formatter = ChatTemplateFormatter.get_formatter("my_formatter", tokenizer)
        tokens = formatter.encode_messages(messages)
    """
    _registry: dict[str, type["ChatTemplateFormatter"]] = {}
    
    def __init__(self, tokenizer, **kwargs):
        """
        Initialize the formatter.
        
        Args:
            tokenizer: HuggingFace tokenizer for encoding
            **kwargs: Additional configuration options
        """
        self.tokenizer = tokenizer
        self.loop = get_event_loop()
    
    @abstractmethod
    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> list[int]:
        """
        Encode messages to token ids.
        
        Args:
            messages: Messages to encode
            add_generation_prompt: Whether to add generation prompt at end
            
        Returns:
            List of token ids
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_system_prompt_tokens(self) -> list[int]:
        """
        Get system prompt token overhead for incremental tokenization.
        
        Returns:
            List of token ids representing the system/prefix portion
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_generation_prompt_tokens(self) -> list[int]:
        """
        Get generation prompt tokens (e.g., '<|assistant|>').
        
        Returns:
            List of token ids for the generation prompt
        """
        raise NotImplementedError
    
    @classmethod
    def get_formatter(cls, name: str, tokenizer, **kwargs) -> "ChatTemplateFormatter":
        """
        Get formatter instance by name.
        
        Args:
            name: Registered formatter name
            tokenizer: Tokenizer for the formatter
            **kwargs: Additional configuration for the formatter
            
        Returns:
            ChatTemplateFormatter instance
            
        Raises:
            ValueError: If formatter name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown chat formatter: {name}. Available: {available}")
        return cls._registry[name](tokenizer, **kwargs)
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a formatter class.
        
        Args:
            name: Name to register the formatter under
            
        Example:
            @ChatTemplateFormatter.register("my_formatter")
            class MyFormatter(ChatTemplateFormatter):
                ...
        """
        def decorator(subclass: type["ChatTemplateFormatter"]) -> type["ChatTemplateFormatter"]:
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def list_formatters(cls) -> list[str]:
        """List all registered formatter names."""
        return list(cls._registry.keys())

