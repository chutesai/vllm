# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)

# Pattern for valid tool call IDs: "functions.name:idx" or "name:idx"
# Function names are word chars and dots (for namespaced functions)
_TOOL_ID_PATTERN = r"(?:functions\.)?(?P<function_name>[\w.]+):(?P<function_idx>\d+)"

# All known Kimi K2 special markers — used for content sanitization
_ALL_MARKERS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_section_begin|>",
    "<|tool_call_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]


def _try_parse_json(s: str) -> str:
    """
    Attempt to parse JSON string, repairing if necessary.
    Returns the (possibly repaired) JSON string, or the original if
    parsing fails entirely.
    """
    s = s.strip()
    if not s:
        return s
    try:
        # Fast path: valid JSON
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Attempt repair: try closing unclosed braces/brackets
    # This handles the most common model errors without adding a dependency
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if open_braces > 0 or open_brackets > 0:
        repaired = s + "]" * open_brackets + "}" * open_braces
        try:
            json.loads(repaired)
            logger.debug(
                "Repaired JSON by closing %d braces, %d brackets",
                open_braces,
                open_brackets,
            )
            return repaired
        except json.JSONDecodeError:
            pass

    # Try stripping trailing comma before closing brace (common model error)
    stripped = re.sub(r",\s*([}\]])", r"\1", s)
    if stripped != s:
        try:
            json.loads(stripped)
            logger.debug("Repaired JSON by removing trailing commas")
            return stripped
        except json.JSONDecodeError:
            pass

    # Combined: strip trailing comma at end of content, then close braces.
    # Handles: '{"a": 1, "b": 2,' → '{"a": 1, "b": 2}'
    if open_braces > 0 or open_brackets > 0:
        # Strip trailing comma (possibly followed by whitespace) at end of string
        combo = re.sub(r",\s*$", "", s)
        combo = combo + "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        try:
            json.loads(combo)
            logger.debug("Repaired JSON by stripping trailing comma and closing braces")
            return combo
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse or repair JSON (len=%d)", len(s))
    return s


def _get_tool_names_from_request(
    request: ChatCompletionRequest | None,
) -> set[str] | None:
    """Extract valid tool/function names from the request, if available."""
    if request is None or not hasattr(request, "tools") or not request.tools:
        return None
    names: set[str] = set()
    for tool in request.tools:
        if hasattr(tool, "function") and hasattr(tool.function, "name"):
            names.add(tool.function.name)
    return names if names else None


def _sanitize_content(content: str) -> str:
    """
    Strip any leaked Kimi K2 special markers from content text.
    This is a safety net — if the regex or streaming logic fails to
    fully consume a marker, we don't want it reaching the user.
    """
    for marker in _ALL_MARKERS:
        if marker in content:
            content = content.replace(marker, "")
    return content


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self._sent_content_idx: int = 0
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        # Section-level state management to prevent token leakage
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        # Buffer size: empirical worst-case for longest marker (~30 chars) * 2
        # + safety margin for unicode + partial overlap. Prevents unbounded growth.
        self.buffer_max_size: int = 1024
        self._buffer_overflow_logged: bool = False  # Log overflow once per session

        # Support both singular and plural variants
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]

        # Individual tool call markers
        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_arg_token: str = "<|tool_call_argument_begin|>"

        # Non-streaming regex: anchored to known end tokens.
        # Captures everything between argument_begin and the next end token
        # (not requiring balanced braces — _try_parse_json handles repair).
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*"
            r"(?P<tool_call_id>" + _TOOL_ID_PATTERN + r")\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>.*?)\s*"
            r"(?:<\|tool_call_end\|>"
            r"|<\|tool_call_begin\|>"
            r"|<\|tool_calls_section_end\|>"
            r"|<\|tool_call_section_end\|>"
            r"|$)",
            re.DOTALL,
        )

        # Streaming regexes: [^<]+ prevents greedy matching across markers
        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>[^<]+:\d+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>.*)",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>[^<]+:\d+)\s*")
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        # Get token IDs for all variants
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure special-token markers appear as literal text in
            # current_text so we can do pure text-based parsing.
            request.skip_special_tokens = False
        return request

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """
        Check for section begin/end markers in text and strip them.
        Returns: (cleaned_text, found_section_begin, found_section_end)
        """
        found_begin = False
        found_end = False
        cleaned = text

        # Check for section begin markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_begin = True

        # Check for section end markers (any variant)
        for variant in self.tool_calls_end_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_end = True
        return cleaned, found_begin, found_end

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self._buffer_overflow_logged = False

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset parent class state
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        logger.debug("Streaming state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        has_tool_section = any(
            variant in model_output for variant in self.tool_calls_start_token_variants
        )
        if not has_tool_section:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # Get valid tool names for validation (if available)
                valid_tools = _get_tool_names_from_request(request)

                tool_calls = []
                for match in self.tool_call_regex.finditer(
                    model_output,
                    timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS,
                ):
                    function_name = match.group("function_name")
                    function_id = match.group("tool_call_id").strip()
                    raw_args = match.group("function_arguments").strip()

                    # Validate function name against request tools
                    if valid_tools and function_name not in valid_tools:
                        logger.warning(
                            "Model called undefined function '%s', "
                            "available tools: %s. Skipping.",
                            function_name,
                            valid_tools,
                        )
                        continue

                    # Attempt to parse/repair JSON arguments
                    function_args = _try_parse_json(raw_args)

                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                logger.debug("Extracted %d tool calls", len(tool_calls))

                # Find the earliest section begin marker
                content_end = len(model_output)
                for variant in self.tool_calls_start_token_variants:
                    idx = model_output.find(variant)
                    if idx != -1 and idx < content_end:
                        content_end = idx
                content = model_output[:content_end]

                # Sanitize content: strip any leaked markers
                content = _sanitize_content(content)

                # If section markers were found but no tool calls matched,
                # still report tools_called=True with empty list so the
                # serving layer knows not to return raw markers as content
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except TimeoutError:
                logger.warning(
                    "Regex timeout in extract_tool_calls after %ds. "
                    "Input length: %d chars.",
                    envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS,
                    len(model_output),
                )
                # Sanitize before returning to avoid marker leakage
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=_sanitize_content(model_output),
                )
            except Exception as e:
                logger.error(
                    "Error in extracting tool call from response: %s", type(e).__name__
                )
                # Sanitize before returning to avoid marker leakage
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=_sanitize_content(model_output),
                )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent content before the tool-calls section, or None.

        Holds back any trailing suffix that partially matches
        ``<|tool_calls_section_begin|>`` to avoid leaking marker bytes.
        """
        if self.tool_calls_start_token not in current_text:
            overlap = partial_tag_overlap(current_text, self.tool_calls_start_token)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = current_text.index(self.tool_calls_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_calls(self, current_text: str) -> list[str]:
        """Extract raw bodies from ``<|tool_call_begin|>…<|tool_call_end|>`` blocks."""
        if self.tool_calls_start_token not in current_text:
            return []

        results: list[str] = []
        pos = current_text.index(self.tool_calls_start_token)
        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            tc_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, tc_start)

            if end != -1:
                tool_call = current_text[tc_start:end]
                pos = end + len(self.tool_call_end_token)
            else:
                tool_call = current_text[tc_start:]
                overlap = partial_tag_overlap(tool_call, self.tool_call_end_token)
                if overlap:
                    tool_call = tool_call[:-overlap]

            results.append(tool_call)

            if end == -1:
                break
        return results

    @staticmethod
    def _extract_tool_id_and_name(
        header: str | None,
    ) -> tuple[str | None, str | None]:
        """Parse ``(tool_id, tool_name)`` from a header
        like ``"functions.get_weather:0"``."""
        if header is None:
            return None, None
        match = re.match(r"(.+:\d+)", header)
        if not match:
            return None, None

        tool_id = match.group(1).strip()
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

    def _split_tool_call(self, tool_call: str) -> tuple[str | None, str | None]:
        """Split a tool-call body into ``(header, arguments)`` at the argument marker.

        Example::
            'get_weather:0 <|tool_call_argument_begin|>{"c'
            -> ("get_weather:0", '{"c')
        """
        arg_pos = tool_call.find(self.tool_call_arg_token)
        if arg_pos == -1:
            return None, None
        header = tool_call[:arg_pos].strip()
        tool_args = tool_call[arg_pos + len(self.tool_call_arg_token) :]
        return header, tool_args

    def _compute_args_diff(self, index: int, tool_args: str | None) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        if tool_args is None:
            return None
        prev = self.streamed_args_for_tool[index]
        if len(tool_args) <= len(prev):
            return None
        diff = tool_args[len(prev) :]
        self.streamed_args_for_tool[index] = tool_args
        self.prev_tool_call_arr[index]["arguments"] = tool_args
        return diff

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        logger.debug("delta_token_ids count: %d", len(delta_token_ids))

        # Auto-reset: if this is the first delta of a new request and we
        # have stale state from a previous (possibly aborted) stream, reset.
        if not previous_text and (self.current_tool_id != -1 or self.in_tool_section):
            logger.warning(
                "Stale streaming state detected at start of new request. "
                "Auto-resetting. (current_tool_id=%d, in_tool_section=%s)",
                self.current_tool_id,
                self.in_tool_section,
            )
            self.reset_streaming_state()

        try:
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            tool_calls = self._extract_tool_calls(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, tool_call in enumerate(tool_calls):
                # First time seeing tool call at index i.
                if i >= len(self.prev_tool_call_arr):
                    # Initialize streaming state.
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                header, tool_args = self._split_tool_call(tool_call)

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if not tool_name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = tool_name
                    self.prev_tool_call_arr[i]["id"] = tool_id
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tool_args)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )
            return None

        except Exception as e:
            logger.error(
                "Error trying to handle streaming tool call: %s", type(e).__name__
            )
            return None  # do not stream a delta. skip this token ID.
