# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class DeepSeekV32ToolParser(ToolParser):
    """
    example tool call content:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.dsml_token: str = "｜DSML｜"
        self.dsml_start_check: str = "<" + self.dsml_token
        self.tool_call_start_token: str = "<｜DSML｜function_calls>"
        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

        # Streaming state variables
        self.current_tool_name_sent: bool = False
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Initialize streaming state variables
        self.current_tool_index: int = 0
        self.invoke_index: int = 0
        self.header_sent: bool = False
        self.current_function_name: str | None = None
        self.current_param_name: str | None = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self.streaming_request: ChatCompletionRequest | None = None

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Flexible regex patterns for complete parsing and streaming
        # Optional DSML marker: matches both "<function_calls>" and "<｜DSML｜function_calls>"
        # Case-insensitive, handles whitespace variations
        _dsml = r"(?:｜\s*DSML\s*｜)?"  # Optional DSML marker
        _tail = r"(?:｜)?\s*>"  # Optional trailing ｜ before >

        self.tool_call_start_regex = re.compile(
            rf"<\s*{_dsml}function_calls{_tail}", re.IGNORECASE
        )
        self.tool_call_end_regex = re.compile(
            rf"</\s*{_dsml}function_calls{_tail}", re.IGNORECASE
        )
        self.invoke_start_regex = re.compile(
            rf'<\s*{_dsml}invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*{_tail}',
            re.IGNORECASE,
        )
        self.invoke_end_regex = re.compile(rf"</\s*{_dsml}invoke{_tail}", re.IGNORECASE)
        self.parameter_start_regex = re.compile(
            rf'<\s*{_dsml}parameter\s+name\s*=\s*["\']([^"\']+)["\'](?:\s+string\s*=\s*["\'](?:true|false)["\'])?\s*{_tail}',
            re.IGNORECASE,
        )
        self.parameter_end_regex = re.compile(
            rf"</\s*{_dsml}parameter{_tail}", re.IGNORECASE
        )

        self.tool_call_complete_regex = re.compile(
            rf"<\s*{_dsml}function_calls{_tail}(.*?)</\s*{_dsml}function_calls{_tail}",
            re.DOTALL | re.IGNORECASE,
        )
        self.invoke_complete_regex = re.compile(
            rf'<\s*{_dsml}invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*{_tail}(.*?)</\s*{_dsml}invoke{_tail}',
            re.DOTALL | re.IGNORECASE,
        )
        # string attribute is now optional
        self.parameter_complete_regex = re.compile(
            rf'<\s*{_dsml}parameter\s+name\s*=\s*["\']([^"\']+)["\'](?:\s+string\s*=\s*["\'](?:true|false)["\'])?\s*{_tail}(.*?)</\s*{_dsml}parameter{_tail}',
            re.DOTALL | re.IGNORECASE,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _get_partial_tool_start(self, text: str) -> str | None:
        """
        Check if text ends with a partial tool call start tag.
        Returns the partial tag if found, None otherwise.

        This handles cases like:
        - "<" (could be start of <function_calls> or <｜DSML｜function_calls>)
        - "<｜DSML｜" (partial DSML tag)
        - "<｜DSML｜function_c" (partial tag)
        - "<function_" (partial non-DSML tag)
        """
        # Possible start patterns to check for partial matches
        patterns = [
            "<｜DSML｜function_calls>",
            "<function_calls>",
        ]

        # Check if text ends with any prefix of these patterns
        for pattern in patterns:
            for i in range(1, len(pattern)):
                prefix = pattern[:i]
                if text.endswith(prefix):
                    return prefix

        return None

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.invoke_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        # Clear previous tool call history to avoid state pollution
        self.prev_tool_call_arr.clear()

    def _parse_invoke_params(self, invoke_str: str) -> dict | None:
        """Parse params from invoke body - supports XML tags or direct JSON."""
        stripped = invoke_str.strip()
        # Try direct JSON first
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Fall back to XML parameter tags
        param_dict = dict()
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        # Use regex for flexible detection
        start_match = self.tool_call_complete_regex.search(model_output)
        if not start_match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []

            # Find all complete tool_call blocks
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                # Find all invokes within this tool_call
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    param_dict = self._parse_invoke_params(invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name.strip(),
                                arguments=json.dumps(param_dict, ensure_ascii=False),
                            ),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            # Extract content before first tool call using regex match position
            content = (
                model_output[: start_match.start()] if start_match.start() > 0 else None
            )

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if (
            name_str.startswith('"')
            and name_str.endswith('"')
            or name_str.startswith("'")
            and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _extract_param_name(self, input_str: str) -> str:
        """Extract param name"""
        start = input_str.find('"') + 1
        end = input_str.find('"', start)
        return input_str[start:end] if start > 0 and end > start else input_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type."""
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Try JSON parse first, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output."""

        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            if delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = len(
                        self.tool_call_start_regex.findall(current_text)
                    ) - len(self.tool_call_end_regex.findall(current_text))
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            invoke_ends = len(self.invoke_end_regex.findall(current_text))
            if invoke_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.in_function = False  # Now we can safely set this to False
                self.accumulated_params = {}
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting - require FULL start tag (with closing >)
            # to avoid false positives on literal text like "<function_calls>" in explanations
            start_match = self.tool_call_start_regex.search(current_text)
            if start_match:
                self.is_tool_call_started = True
                # Return any content before the tool call
                delta_start_match = self.tool_call_start_regex.search(delta_text)
                if delta_start_match:
                    content_before = delta_text[: delta_start_match.start()]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Check if we're between tool calls - skip whitespace
                stripped_current = current_text.rstrip()
                end_match = self.tool_call_end_regex.search(stripped_current)
                if (
                    end_match
                    and end_match.end() == len(stripped_current)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                # Check if current_text ends with a partial tool call start tag
                # to avoid leaking partial tags as content
                partial_tag = self._get_partial_tool_start(current_text)
                if partial_tag:
                    # Don't emit the partial tag as content yet
                    if len(delta_text) <= len(partial_tag):
                        # Entire delta is part of partial tag, emit nothing
                        return None
                    else:
                        # Emit content before the partial tag
                        return DeltaMessage(content=delta_text[: -len(partial_tag)])
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        invoke_start_matches = list(self.invoke_start_regex.finditer(current_text))
        if self.current_tool_index >= len(invoke_start_matches):
            # We're past all tool calls, shouldn't be here
            return None

        # Find the current tool call portion
        invoke_start_match = invoke_start_matches[self.current_tool_index]
        invoke_start_idx = invoke_start_match.start()
        invoke_start_end = invoke_start_match.end()
        # Find where this tool call ends (or current position if not ended yet)
        invoke_end_match = self.invoke_end_regex.search(current_text, invoke_start_end)
        if invoke_end_match is None:
            tool_text = current_text[invoke_start_idx:]
        else:
            tool_text = current_text[invoke_start_idx : invoke_end_match.end()]

        # Looking for function header
        if not self.header_sent:
            function_name_raw = invoke_start_match.group(1)
            self.current_function_name = function_name_raw.strip()
            self.current_tool_id = self._generate_tool_call_id()
            self.header_sent = True
            self.in_function = True

            # Add to prev_tool_call_arr immediately when we detect a tool call
            # Each tool call should be recorded regardless of function name
            # Ensure we don't add the same tool call index multiple times
            if len(self.prev_tool_call_arr) <= self.current_tool_index:
                self.prev_tool_call_arr.append(
                    {
                        "name": self.current_function_name,
                        "arguments": "{}",  # Placeholder, will be updated later
                    }
                )

            # Send header with function info
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(
                            name=self.current_function_name, arguments=""
                        ),
                        type="function",
                    )
                ]
            )

        # We've sent header, now handle function body
        if self.in_function:
            # Send opening brace if not sent yet
            if self.in_function and not self.json_started:
                self.json_started = True
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            # Make sure json_started is set if we're processing parameters
            if not self.json_started:
                self.json_started = True

            # Check for function end in accumulated text
            if not self.json_closed and self.invoke_end_regex.search(tool_text):
                # Count total parameters in the tool text
                total_param_count = len(self.parameter_start_regex.findall(tool_text))

                # Only close JSON if all parameters have been processed
                if self.param_count >= total_param_count:
                    # Close JSON
                    self.json_closed = True

                    # Extract complete tool call
                    # Find the invoke content
                    invoke_start_match = self.invoke_start_regex.search(tool_text)
                    invoke_end_match = self.invoke_end_regex.search(tool_text)
                    if invoke_start_match and invoke_end_match:
                        invoke_content = tool_text[
                            invoke_start_match.end() : invoke_end_match.start()
                        ]
                        # Parse to get the complete arguments
                        try:
                            invoke_params = self._parse_invoke_params(invoke_content)
                            if invoke_params and self.current_tool_index < len(
                                self.prev_tool_call_arr
                            ):
                                # Update existing entry in prev_tool_call_arr
                                self.prev_tool_call_arr[self.current_tool_index][
                                    "arguments"
                                ] = json.dumps(invoke_params, ensure_ascii=False)
                        except Exception:
                            pass  # Ignore parsing errors during streaming

                    result = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments="}"),
                            )
                        ]
                    )

                    # Reset state for next tool
                    self.json_closed = True
                    self.in_function = False
                    self.accumulated_params = {}

                    logger.debug("[M2_STREAMING] Tool call completed")

                    return result
                else:
                    # Don't close JSON yet, continue processing parameters
                    return None

            # Look for parameters
            # Find all parameter starts
            param_start_matches = list(self.parameter_start_regex.finditer(tool_text))

            # Check if we should start a new parameter
            if (
                not self.in_param
                and self.param_count < len(param_start_matches)
                and len(param_start_matches) > self.param_count
            ):
                # Process the next parameter
                param_start_match = param_start_matches[self.param_count]
                self.current_param_name = param_start_match.group(1)
                value_start = param_start_match.end()
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                # Find where this parameter ends
                param_end_match = self.parameter_end_regex.search(value_text)
                param_end_idx = param_end_match.start() if param_end_match else -1
                if param_end_idx == -1:
                    # No closing tag, look for next parameter or function end
                    next_param_match = self.parameter_start_regex.search(value_text)
                    func_end_match = self.invoke_end_regex.search(value_text)
                    next_param_idx = (
                        next_param_match.start() if next_param_match else -1
                    )
                    func_end_idx = func_end_match.start() if func_end_match else -1

                    if next_param_idx != -1 and (
                        func_end_idx == -1 or next_param_idx < func_end_idx
                    ):
                        param_end_idx = next_param_idx
                    elif func_end_idx != -1:
                        param_end_idx = func_end_idx
                    else:
                        # Neither found, check if tool call is complete
                        if self.invoke_end_regex.search(tool_text):
                            # Tool call and parameter is complete
                            param_end_idx = len(value_text)
                        else:
                            # Still streaming, wait for more content
                            return None

                if param_end_idx != -1:
                    # Complete parameter found
                    param_value = value_text[:param_end_idx]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                    # Store raw value for later processing
                    self.accumulated_params[self.current_param_name] = param_value

                    # Get parameter configuration for type conversion
                    param_config = {}
                    if self.streaming_request and self.streaming_request.tools:
                        for tool in self.streaming_request.tools:
                            if (
                                hasattr(tool, "function")
                                and tool.function.name == self.current_function_name
                                and hasattr(tool.function, "parameters")
                            ):
                                params = tool.function.parameters
                                if isinstance(params, dict) and "properties" in params:
                                    param_config = params["properties"]
                                break

                    # Get parameter type
                    param_type = "string"
                    if (
                        self.current_param_name in param_config
                        and isinstance(param_config[self.current_param_name], dict)
                        and "type" in param_config[self.current_param_name]
                    ):
                        param_type = param_config[self.current_param_name]["type"]

                    # Convert param value to appropriate type
                    converted_value = self._convert_param_value(param_value, param_type)

                    # Build JSON fragment based on the converted type
                    # Use json.dumps to properly serialize the value
                    serialized_value = json.dumps(converted_value, ensure_ascii=False)

                    if self.param_count == 0:
                        json_fragment = (
                            f'"{self.current_param_name}": {serialized_value}'
                        )
                    else:
                        json_fragment = (
                            f', "{self.current_param_name}": {serialized_value}'
                        )

                    self.param_count += 1

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments=json_fragment),
                            )
                        ]
                    )

        return None
