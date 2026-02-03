"""
Tool parser module for extracting tool calls from LLM responses.

This module provides:
- ToolParser: Abstract base class for tool call parsers
- FunctionCall: Data class for representing parsed function calls
- Built-in parsers: PythonCodeParser for ```python``` code blocks

Design aligned with verl's tool_parser.py and slime's tools.py patterns.
"""

import re
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """
    Represents a parsed function call from LLM response.
    
    Attributes:
        name: The name of the function/tool to call
        arguments: JSON string of arguments for the function call
    """
    name: str
    arguments: str  # JSON string
    
    def get_arguments_dict(self) -> dict[str, Any]:
        """Parse arguments JSON string to dictionary"""
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse arguments: {self.arguments}")
            return {}


class ToolParser(ABC):
    """
    Abstract base class for tool call parsers.
    
    Provides a registry pattern for extensibility, allowing different
    tool call formats (code blocks, function call syntax, etc.) to be
    parsed according to model-specific patterns.
    
    Usage:
        @ToolParser.register("python_code")
        class PythonCodeParser(ToolParser):
            ...
        
        parser = ToolParser.get_parser("python_code", tokenizer)
        content, calls = await parser.extract_tool_calls(response_ids)
    """
    _registry: dict[str, type["ToolParser"]] = {}
    
    def __init__(self, tokenizer=None):
        """
        Initialize the parser.
        
        Args:
            tokenizer: Optional tokenizer for decoding token ids to text.
                       If None, text-based extraction methods should be used.
        """
        self.tokenizer = tokenizer
    
    @abstractmethod
    async def extract_tool_calls(
        self, 
        response: str | list[int]
    ) -> tuple[str, list[FunctionCall]]:
        """
        Extract tool calls from the response.
        
        Args:
            response: Either a string response or list of token ids.
                     If token ids are provided, the tokenizer will be used to decode.
        
        Returns:
            A tuple of (remaining_content, list_of_function_calls):
            - remaining_content: The text content with tool calls removed
            - list_of_function_calls: Extracted FunctionCall objects
        """
        raise NotImplementedError
    
    def _decode_if_needed(self, response: str | list[int]) -> str:
        """Convert token ids to string if needed"""
        if isinstance(response, str):
            return response
        if self.tokenizer is None:
            raise ValueError("Tokenizer required to decode token ids")
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    @classmethod
    def get_parser(cls, name: str, tokenizer=None) -> "ToolParser":
        """
        Get a parser instance by name.
        
        Args:
            name: Registered parser name
            tokenizer: Tokenizer for the parser
            
        Returns:
            ToolParser instance
            
        Raises:
            ValueError: If parser name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown tool parser: {name}. Available: {available}")
        return cls._registry[name](tokenizer)
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a parser class.
        
        Args:
            name: Name to register the parser under
            
        Example:
            @ToolParser.register("my_parser")
            class MyParser(ToolParser):
                ...
        """
        def decorator(subclass: type["ToolParser"]) -> type["ToolParser"]:
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def list_parsers(cls) -> list[str]:
        """List all registered parser names"""
        return list(cls._registry.keys())


@ToolParser.register("python_code")
class PythonCodeParser(ToolParser):
    """
    Parser for Python code blocks in LLM responses.
    
    Extracts code from ```python...``` or ```...``` blocks and
    creates FunctionCall objects targeting 'code_interpreter' tool.
    
    This aligns with the code_interpreter tool registered in tools.py.
    """
    
    # Pattern for matching Python code blocks
    CODE_BLOCK_PATTERN = re.compile(r"```(?:python\n)?(.*?)```", re.DOTALL)
    
    async def extract_tool_calls(
        self, 
        response: str | list[int]
    ) -> tuple[str, list[FunctionCall]]:
        """
        Extract Python code blocks as tool calls.
        
        Args:
            response: LLM response text or token ids
            
        Returns:
            Tuple of (content_without_code_blocks, list_of_code_interpreter_calls)
        """
        text = self._decode_if_needed(response)
        
        matches = self.CODE_BLOCK_PATTERN.findall(text)
        function_calls = []
        
        for match in matches:
            code = match.strip()
            if code:  # Only add non-empty code blocks
                function_calls.append(FunctionCall(
                    name="code_interpreter",
                    arguments=json.dumps({"code": code}, ensure_ascii=False)
                ))
        
        # Remove code blocks from content
        content = self.CODE_BLOCK_PATTERN.sub("", text).strip()
        
        return content, function_calls


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """
    Parser for Hermes-style tool calls.
    
    Hermes format uses <tool_call>...</tool_call> tags with JSON content.
    Adapted from verl's HermesToolParser.
    """
    
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    
    async def extract_tool_calls(
        self, 
        response: str | list[int]
    ) -> tuple[str, list[FunctionCall]]:
        """
        Extract Hermes-style tool calls.
        
        Expected format:
            <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
        """
        text = self._decode_if_needed(response)
        
        if self.TOOL_CALL_START not in text or self.TOOL_CALL_END not in text:
            return text, []
        
        matches = self.TOOL_CALL_PATTERN.findall(text)
        function_calls = []
        
        for match in matches:
            try:
                call_data = json.loads(match.strip())
                name = call_data.get("name")
                arguments = call_data.get("arguments", {})
                
                if name:
                    function_calls.append(FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False) 
                                  if isinstance(arguments, dict) else str(arguments)
                    ))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {e}")
        
        # Remove tool call blocks from content
        content = self.TOOL_CALL_PATTERN.sub("", text).strip()
        
        return content, function_calls


# Convenience function for quick extraction
async def extract_code_blocks(text: str) -> list[str]:
    """
    Quick helper to extract all Python code blocks from text.
    
    Args:
        text: Input text containing code blocks
        
    Returns:
        List of code strings extracted from blocks
    """
    parser = PythonCodeParser()
    _, calls = await parser.extract_tool_calls(text)
    return [call.get_arguments_dict().get("code", "") for call in calls]


def extract_code_blocks_sync(text: str) -> list[str]:
    """
    Synchronous version of extract_code_blocks.
    
    Args:
        text: Input text containing code blocks
        
    Returns:
        List of code strings extracted from blocks
    """
    matches = PythonCodeParser.CODE_BLOCK_PATTERN.findall(text)
    return [m.strip() for m in matches if m.strip()]
