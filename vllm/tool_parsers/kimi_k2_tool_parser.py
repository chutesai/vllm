# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

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

    logger.warning("Could not parse or repair JSON: %s", s[:200])
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
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

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

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

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
            except Exception:
                logger.exception("Error in extracting tool call from response.")
                # Sanitize before returning to avoid marker leakage
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=_sanitize_content(model_output),
                )

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
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

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

        # Flag to defer section exit until after tool parsing completes
        deferred_section_exit = False

        # Add delta to buffer for split marker detection
        self.token_buffer += delta_text

        # Enforce buffer size limit to prevent memory issues
        if len(self.token_buffer) > self.buffer_max_size:
            if not self._buffer_overflow_logged:
                logger.warning(
                    "Token buffer exceeded max size (%d bytes), flushing excess. "
                    "This may indicate very long markers or unusual tokenization.",
                    self.buffer_max_size,
                )
                self._buffer_overflow_logged = True
            # Keep only the most recent content that might contain partial markers
            self.token_buffer = self.token_buffer[-self.buffer_max_size // 2 :]

        # Check buffer for section markers (handles split tokens)
        buffered_text, found_section_begin, found_section_end = (
            self._check_and_strip_markers(self.token_buffer)
        )

        # Track section state transitions
        if found_section_begin and not self.in_tool_section:
            logger.debug("Entering tool section")
            self.in_tool_section = True
            self.token_buffer = buffered_text  # Use cleaned buffer

        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            # CRITICAL: Don't exit early if tool_call_end is in this chunk.
            # Tool parser must emit final arguments/close first to avoid dropping
            # the final tool update and leaking tokens into reasoning channel.
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                # Defer exit until after tool parsing completes
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
                self.token_buffer = buffered_text
            else:
                # No tool call ending, safe to exit immediately
                logger.debug("Exiting tool section")
                self._reset_section_state()
                # Extract any content AFTER the section end marker in delta_text
                # (don't use buffered_text as it contains tool call data)
                post_section_content = ""
                for variant in self.tool_calls_end_token_variants:
                    if variant in delta_text:
                        parts = delta_text.split(variant, 1)
                        if len(parts) > 1:
                            post_section_content = parts[1]
                        break
                # Sanitize any leaked markers in post-section content
                post_section_content = _sanitize_content(post_section_content)
                if post_section_content.strip():
                    return DeltaMessage(content=post_section_content)
                return DeltaMessage(content="")
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )

        # Early return: if no section token detected yet, return as reasoning content
        if not has_section_token and not self.in_tool_section:
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected above
            return DeltaMessage(content=delta_text)

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing to ensure markers are removed cleanly before pattern matching.
        # No double-stripping occurs because section markers and tool call
        # markers are distinct.
        delta_text, _, _ = self._check_and_strip_markers(delta_text)

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                # Suppress content between section begin and first tool begin
                # (header noise). Don't suppress content between tools to avoid
                # breaking potential delimiter characters.
                if self.in_tool_section and cur_tool_start_count == 0:
                    logger.debug(
                        "In tool section before first tool, suppressing: %s",
                        delta_text,
                    )
                    # Return empty delta to maintain iterator contract
                    return DeltaMessage(content="")
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1]
                    .split(self.tool_call_end_token)[0]
                    .rstrip()
                )
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()

            # case -- we're starting a new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[
                        -1
                    ]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None

                # Extract full arguments from tool_call_portion and diff
                # against what we've already streamed
                full_arguments = None
                if tool_call_portion:
                    tc_match = self.stream_tool_call_portion_regex.match(
                        tool_call_portion
                    )
                    if tc_match:
                        full_arguments = tc_match.group("function_arguments")

                already_streamed = (
                    self.streamed_args_for_tool[self.current_tool_id]
                    if self.current_tool_id < len(self.streamed_args_for_tool)
                    else ""
                )

                if full_arguments and full_arguments.startswith(already_streamed):
                    remaining = full_arguments[len(already_streamed) :]
                    if remaining:
                        logger.debug(
                            "Finishing tool and found diff that had not "
                            "been streamed yet: %s",
                            remaining,
                        )
                        self.streamed_args_for_tool[self.current_tool_id] = (
                            full_arguments
                        )
                        # Handle deferred section exit before returning
                        if deferred_section_exit and self.in_tool_section:
                            logger.debug("Completing deferred section exit")
                            self._reset_section_state()
                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=remaining
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )

                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return None

            # case -- otherwise we're just generating text
            else:
                # Check if we're in tool section - if so, suppress
                if self.in_tool_section:
                    logger.debug("In tool section, suppressing text generation")
                    # Handle deferred section exit before returning
                    if deferred_section_exit:
                        self._reset_section_state()
                    return DeltaMessage(content="")
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(":")[0].split(".")[-1]
                    current_tool_call["id"] = tool_id.strip()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = (
                        self.stream_tool_call_name_regex.match(tool_call_portion)
                    )
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(":")[0].split(".")[-1]
                        current_tool_call["id"] = tool_id_str.strip()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                # CRITICAL: Never return content if we're in a tool section
                if self.in_tool_section:
                    return None
                delta = (
                    DeltaMessage(content=delta_text)
                    if text_portion is not None
                    else None
                )
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=cur_arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    logger.debug("got diff %s", delta_text)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            # Handle deferred section exit after tool parsing completes
            if deferred_section_exit and self.in_tool_section:
                logger.debug("Completing deferred section exit")
                self._reset_section_state()

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
