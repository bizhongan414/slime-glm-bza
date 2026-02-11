"""
Chat template formatter module for handling model-specific message encoding.

This module provides:
- ChatTemplateFormatter: Abstract base class for chat template formatting

Formatters are loaded via full class path (e.g. 
examples.code.tool_utils.deepseek_v32_formatter.DeepSeekV32Formatter).
"""

import importlib
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
        formatter = ChatTemplateFormatter.get_formatter(
            "examples.code.tool_utils.deepseek_v32_formatter.DeepSeekV32Formatter",
            tokenizer,
            thinking_mode="thinking"
        )
        tokens = formatter.encode_messages(messages)
    """
    
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
    def get_formatter(cls, class_path: str, tokenizer, **kwargs) -> "ChatTemplateFormatter":
        """
        Get formatter instance by full class path, using dynamic import.
        
        Args:
            class_path: Full qualified class path 
                        (e.g. "examples.code.tool_utils.deepseek_v32_formatter.DeepSeekV32Formatter")
            tokenizer: Tokenizer for the formatter
            **kwargs: Additional configuration for the formatter
            
        Returns:
            ChatTemplateFormatter instance
            
        Raises:
            ValueError: If class cannot be imported or is not a valid formatter
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            formatter_cls = getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Cannot load chat formatter '{class_path}': {e}. "
                f"Ensure the class path is fully qualified, e.g. "
                f"'examples.code.tool_utils.deepseek_v32_formatter.DeepSeekV32Formatter'"
            ) from e
        
        if not issubclass(formatter_cls, ChatTemplateFormatter):
            raise ValueError(
                f"'{class_path}' is not a subclass of ChatTemplateFormatter"
            )
        
        return formatter_cls(tokenizer, **kwargs)
