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
from examples.code.global_utils import get_event_loop

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
        self.loop = get_event_loop()

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
    
    async def _decode_if_needed(self, response: str | list[int]) -> str:
        """Convert token ids to string if needed (async, non-blocking)."""
        if isinstance(response, str):
            return response
        if self.tokenizer is None:
            raise ValueError("Tokenizer required to decode token ids")
        # Use run_in_executor to avoid blocking the event loop
        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response, skip_special_tokens=True)
        )
    
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
        text = await self._decode_if_needed(response)
        
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
        text = await self._decode_if_needed(response)
        
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


@ToolParser.register("deepseek_dsml")
class DeepSeekDSMLParser(ToolParser):
    """
    Parser for DeepSeek V3.2 DSML tool call format.
    
    DSML format uses ｜DSML｜ tags for structured tool calls:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="function_name">
    <｜DSML｜parameter name="param" string="true">value</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    
    Based on encoding_dsv32.py reference implementation.
    """
    
    DSML_TOKEN = "｜DSML｜"
    
    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Regex patterns for parsing
        self.function_calls_pattern = re.compile(
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", 
            re.DOTALL
        )
        self.invoke_pattern = re.compile(
            r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', 
            re.DOTALL
        )
        self.parameter_pattern = re.compile(
            r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', 
            re.DOTALL
        )
    
    async def extract_tool_calls(
        self, 
        response: str | list[int]
    ) -> tuple[str, list[FunctionCall]]:
        """
        Extract DeepSeek DSML-style tool calls.
        
        Args:
            response: LLM response text or token ids
            
        Returns:
            Tuple of (content_without_tool_calls, list_of_function_calls)
        """
        breakpoint()
        text = await self._decode_if_needed(response)
        
        # Check if there are any function calls
        if f"<{self.DSML_TOKEN}function_calls>" not in text:
            return text, []
        
        function_calls = []
        
        # Find all function_calls blocks
        blocks = self.function_calls_pattern.findall(text)
        
        for block in blocks:
            # Find all invoke elements in this block
            invokes = self.invoke_pattern.findall(block)
            
            for tool_name, invoke_content in invokes:
                # Parse parameters
                params = self.parameter_pattern.findall(invoke_content)
                arguments = {}
                breakpoint()
                for param_name, is_string, param_value in params:
                    if is_string == "true":
                        # String value - use as-is
                        arguments[param_name] = param_value
                    else:
                        # Non-string value - parse as JSON
                        try:
                            arguments[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            # Fallback to string if JSON parsing fails
                            logger.warning(f"Failed to parse parameter as JSON: {param_value}")
                            arguments[param_name] = param_value
                
                function_calls.append(FunctionCall(
                    name=tool_name,
                    arguments=json.dumps(arguments, ensure_ascii=False)
                ))
        
        # Remove function_calls blocks from content
        content = self.function_calls_pattern.sub("", text).strip()
        
        return content, function_calls


@ToolParser.register("qwen3")
class Qwen3JSONParser(ToolParser):
    """
    Parser for Qwen3 series (JSON inside XML format).
    Matches the official chat_template output.
    """
    
    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # 1. 只需要匹配最外层的 <tool_call> 标签
        self.tool_call_pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>", 
            re.DOTALL
        )

    async def extract_tool_calls(
        self, 
        response: str | list[int]
    ) -> tuple[str, list[FunctionCall]]:
        
        text = await self._decode_if_needed(response)
        function_calls = []
        
        # 查找所有 <tool_call> 内容
        tool_call_blocks = self.tool_call_pattern.findall(text)
        
        for block in tool_call_blocks:
            try:
                # 2. Qwen3 的内容是纯 JSON，直接 load
                # block 可能是 '{"name": "func", "arguments": {...}}'
                tool_data = json.loads(block.strip())
                
                # 兼容可能的不同 JSON 结构，通常是 standard format
                func_name = tool_data.get("name")
                arguments = tool_data.get("arguments")
                
                # 有些时候 arguments 已经是 dict，有些时候是 string，视情况处理
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                
                if func_name:
                    function_calls.append(FunctionCall(
                        name=func_name,
                        arguments=arguments
                    ))
            except json.JSONDecodeError:
                # 容错处理：模型生成的 JSON 可能不合法
                print(f"Warning: Failed to decode JSON tool call: {block}")
                continue

        # 3. 清理文本：移除所有工具调用标签，保留纯文本对话（包括 <think>）
        content = self.tool_call_pattern.sub("", text).strip()
        
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

