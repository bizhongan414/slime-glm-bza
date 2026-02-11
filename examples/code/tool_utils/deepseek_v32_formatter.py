"""
DeepSeek V3.2 Chat Template Formatter with DSML tool call format.

This module provides:
- DeepSeekV32Formatter: Formatter for DeepSeek V3.2's special DSML format
- Supports thinking mode with <think></think> tags
- Handles ｜DSML｜ format for tool calls

Based on encoding_dsv32.py reference implementation.
"""

import copy
import json
import logging
from typing import Any, Optional

from examples.code.tool_utils.chat_formatter import ChatTemplateFormatter

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Templates (from encoding_dsv32.py)
# ============================================================================

TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

**Important: Tool Usage Guidelines**
- You should ALWAYS prefer using tools over reasoning alone when the task involves computation, data processing, or verification.
- Do NOT attempt to mentally compute, simulate, or guess results that a tool can provide accurately. Use the tool instead.
- When solving problems, write code to verify your approach rather than relying solely on your reasoning.
- If a task can be broken into steps, use tools at each step to validate intermediate results.
- Even if you are confident in your reasoning, use tools to double-check your answer when possible.

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}function_calls>
...
</{dsml_token}function_calls>

<function_results>
...
</function_results>

{thinking_start_token}...thinking about results{thinking_end_token}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""


def _to_json(value: Any) -> str:
    """Convert value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def _tools_from_openai_format(tools: list[dict]) -> list[dict]:
    """Extract function definitions from OpenAI tool format."""
    return [tool["function"] for tool in tools]


def _tool_calls_from_openai_format(tool_calls: list[dict]) -> list[dict]:
    """Convert OpenAI tool call format to simple format."""
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


class DeepSeekV32Formatter(ChatTemplateFormatter):
    """
    DeepSeek V3.2 specific formatter with DSML tool call format.
    
    Supports:
    - DSML format for tool calls (｜DSML｜ tags)
    - Thinking mode with <think></think> tags
    - Incremental encoding with context
    
    Based on encoding_dsv32.py reference implementation.
    """
    
    # Special tokens
    BOS_TOKEN = "<｜begin▁of▁sentence｜>"
    EOS_TOKEN = "<｜end▁of▁sentence｜>"
    THINKING_START = "<think>"
    THINKING_END = "</think>"
    DSML_TOKEN = "｜DSML｜"
    
    # Message templates
    SYSTEM_MSG_TEMPLATE = "{content}"
    USER_MSG_TEMPLATE = "<｜User｜>{content}<｜Assistant｜>"
    ASSISTANT_MSG_TEMPLATE = "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"
    THINKING_TEMPLATE = "{reasoning_content}"
    
    # Tool templates
    TOOL_CALL_TEMPLATE = "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>"
    TOOL_CALLS_TEMPLATE = "<{dsml_token}function_calls>\n{tool_calls}\n</{dsml_token}function_calls>"
    TOOL_OUTPUT_TEMPLATE = "\n<result>{content}</result>"
    RESPONSE_FORMAT_TEMPLATE = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
    
    def __init__(
        self, 
        tokenizer, 
        thinking_mode: str = "chat",
        add_default_bos_token: bool = True,
        **kwargs
    ):
        """
        Initialize DeepSeek V3.2 formatter.
        
        Args:
            tokenizer: HuggingFace tokenizer
            thinking_mode: "thinking" or "chat", controls whether to output thinking tags
            add_default_bos_token: Whether to add BOS token at the start
            **kwargs: Additional options
        """
        super().__init__(tokenizer, **kwargs)
        self.thinking_mode = thinking_mode
        self.add_default_bos_token = add_default_bos_token
        
        # Pre-calculate tokens
        self._system_prompt_tokens = self._calculate_system_prompt_tokens()
        self._generation_prompt_tokens = self._extract_generation_prompt_tokens()
    
    def encode_messages(
        self,
        messages: list[dict[str, Any]],
        context: Optional[list[dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> list[int]:
        """
        Encode messages to token ids using DeepSeek V3.2 format.
        
        Args:
            messages: New messages to encode
            context: Previous messages (for incremental tokenization)
            add_generation_prompt: Whether to add generation prompt at end
            
        Returns:
            List of token ids
        """
        breakpoint()
        context = context if context else []
        full_messages = context + messages
        
        # Build prompt string
        prompt = self.BOS_TOKEN if self.add_default_bos_token and len(context) == 0 else ""
        
        # Render only the new messages (with context for reference)
        for idx in range(len(messages)):
            prompt += self._render_message(idx + len(context), full_messages)
        
        # Encode to token ids
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tokens
    
    def get_system_prompt_tokens(self) -> list[int]:
        """Get pre-calculated system prompt tokens."""
        return self._system_prompt_tokens
    
    def get_generation_prompt_tokens(self) -> list[int]:
        """Get pre-calculated generation prompt tokens."""
        return self._generation_prompt_tokens
    
    def _calculate_system_prompt_tokens(self) -> list[int]:
        """Calculate system prompt tokens for incremental tokenization."""
        # For DeepSeek, the BOS token is the system prompt
        return self.tokenizer.encode(self.BOS_TOKEN, add_special_tokens=False)
    
    def _extract_generation_prompt_tokens(self) -> list[int]:
        """Extract generation prompt tokens."""
        # For DeepSeek in thinking mode, generation prompt includes <think>
        if self.thinking_mode == "thinking":
            return self.tokenizer.encode(self.THINKING_START, add_special_tokens=False)
        else:
            return self.tokenizer.encode(self.THINKING_END, add_special_tokens=False)
    
    def _find_last_user_index(self, messages: list[dict[str, Any]]) -> int:
        """Find the index of the last user message."""
        last_user_index = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") in ["user", "developer"]:
                last_user_index = idx
                break
        return last_user_index
    
    def _render_tools(self, tools: list[dict[str, Any]]) -> str:
        """Render tools as system message content."""
        tools_json = [_to_json(t) for t in tools]
        return TOOLS_SYSTEM_TEMPLATE.format(
            tool_schemas="\n".join(tools_json),
            dsml_token=self.DSML_TOKEN,
            thinking_start_token=self.THINKING_START,
            thinking_end_token=self.THINKING_END,
        )
    
    def _encode_arguments_to_dsml(self, tool_call: dict[str, str]) -> str:
        """Encode tool call arguments to DSML format."""
        param_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
        param_strs = []
        
        arguments = json.loads(tool_call["arguments"])
        
        for k, v in arguments.items():
            param_str = param_template.format(
                dsml_token=self.DSML_TOKEN,
                key=k,
                is_str="true" if isinstance(v, str) else "false",
                value=v if isinstance(v, str) else _to_json(v),
            )
            param_strs.append(param_str)
        
        return "\n".join(param_strs)
    
    def _render_message(self, index: int, messages: list[dict[str, Any]]) -> str:
        """Render a single message at the given index."""
        assert 0 <= index < len(messages)
        assert self.thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{self.thinking_mode}`"
        
        prompt = ""
        msg = messages[index]
        last_user_idx = self._find_last_user_index(messages)
        
        role = msg.get("role")
        content = msg.get("content")
        tools = msg.get("tools")
        response_format = msg.get("response_format")
        tool_calls = msg.get("tool_calls")
        reasoning_content = msg.get("reasoning_content")
        
        if tools:
            tools = _tools_from_openai_format(tools)
        if tool_calls:
            tool_calls = _tool_calls_from_openai_format(tool_calls)
        
        if role == "system":
            prompt += self.SYSTEM_MSG_TEMPLATE.format(content=content or "")
            if tools:
                prompt += "\n\n" + self._render_tools(tools)
            if response_format:
                prompt += "\n\n" + self.RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(response_format))
        
        elif role == "developer":
            assert content, f"Invalid message for role `{role}`: {msg}"
            content_developer = ""
            if tools:
                content_developer += "\n\n" + self._render_tools(tools)
            if response_format:
                content_developer += "\n\n" + self.RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(response_format))
            content_developer += "\n\n# The user's message is: {}".format(content)
            
            prompt += self.USER_MSG_TEMPLATE.format(content=content_developer)
            if index == last_user_idx and self.thinking_mode == "thinking":
                prompt += self.THINKING_START
            else:
                prompt += self.THINKING_END
        
        elif role == "user":
            prompt += self.USER_MSG_TEMPLATE.format(content=content)
            if index == last_user_idx and self.thinking_mode == "thinking":
                prompt += self.THINKING_START
            else:
                prompt += self.THINKING_END
        
        elif role == "tool":
            # Simplified logic: determine position based on tool messages in current batch
            # This allows rendering without needing context (previous assistant message)
            
            # Count consecutive tool messages ending at current index
            breakpoint()
            tool_messages_end = index
            tool_messages_start = index
            while tool_messages_start > 0 and messages[tool_messages_start - 1].get("role") == "tool":
                tool_messages_start -= 1
            
            # Calculate position within this tool batch
            tool_order = index - tool_messages_start + 1  # 1-indexed
            tool_count = tool_messages_end - tool_messages_start + 1
            
            # Check if there are more tool messages after current
            while tool_messages_end < len(messages) - 1 and messages[tool_messages_end + 1].get("role") == "tool":
                tool_messages_end += 1
                tool_count = tool_messages_end - tool_messages_start + 1
            
            if tool_order == 1:
                prompt += "\n\n<function_results>"
            
            prompt += self.TOOL_OUTPUT_TEMPLATE.format(content=content)
            
            if tool_order == tool_count:
                prompt += "\n</function_results>"
                # Add generation prompt after tool results (matching encoding_dsv32.py)
                if index >= last_user_idx and self.thinking_mode == "thinking":
                    prompt += "\n\n" + self.THINKING_START
                else:
                    prompt += "\n\n" + self.THINKING_END

        elif role == "assistant":
            thinking_part = ""
            tool_calls_content = ""
            
            if tool_calls:
                rendered_calls = [
                    self.TOOL_CALL_TEMPLATE.format(
                        dsml_token=self.DSML_TOKEN,
                        name=tc.get("name"),
                        arguments=self._encode_arguments_to_dsml(tc)
                    )
                    for tc in tool_calls
                ]
                tool_calls_content += "\n\n" + self.TOOL_CALLS_TEMPLATE.format(
                    dsml_token=self.DSML_TOKEN,
                    tool_calls="\n".join(rendered_calls)
                )
            
            summary_content = content or ""
            
            if self.thinking_mode == "thinking" and index > last_user_idx:
                assert reasoning_content or tool_calls, \
                    f"ThinkingMode: {self.thinking_mode}, invalid message without reasoning_content/tool_calls `{msg}` after last user message"
                thinking_part = self.THINKING_TEMPLATE.format(reasoning_content=reasoning_content or "") + self.THINKING_END
            
            prompt += self.ASSISTANT_MSG_TEMPLATE.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tool_calls_content,
            )
        else:
            raise NotImplementedError(f"Unknown role: {role}")
        
        return prompt
    
