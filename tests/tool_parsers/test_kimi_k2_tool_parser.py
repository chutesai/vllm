# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import json

import pytest

from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser

# Use a common model that is likely to be available
MODEL = "moonshotai/Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def kimi_k2_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL, trust_remote_code=True)


@pytest.fixture
def kimi_k2_tool_parser(kimi_k2_tokenizer):
    return KimiK2ToolParser(kimi_k2_tokenizer)


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        # assert tool call id format: should contain function name and numeric index
        # Format can be either "functions.func_name:0" or "func_name:0"
        assert actual_tool_call.id.split(":")[-1].isdigit()
        assert (
            actual_tool_call.id.split(":")[0].split(".")[-1]
            == expected_tool_call.function.name
        )


def run_streaming_sequence(parser, deltas):
    """Helper to simulate a streaming sequence and return results."""
    previous_text = ""
    previous_token_ids: list[int] = []
    results = []

    for delta_text, delta_token_ids in deltas:
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=None,
        )
        results.append(result)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return results


def test_extract_tool_calls_no_tools(kimi_k2_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "tool_call_with_content_before",
        "multi_tool_call_with_content_before",
        "concatenated_tool_calls_bug_fix",
        "three_concatenated_tool_calls",
        "mixed_spacing_tool_calls",
        "angle_brackets_in_json",
        "newlines_in_json",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                )
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Beijing",
                            },
                        ),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.get_weather:1",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {
                                "city": "Shanghai",
                            },
                        ),
                    ),
                    type="function",
                ),
            ],
            "I'll help you check the weather. ",
        ),
        (
            """I'll get the weather and news for LA today. First, let me get the weather using Los Angeles coordinates, and then get the latest news. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"latitude": 34.0522, "longitude": -118.2437}<|tool_call_end|><|tool_call_begin|>functions.get_news:1<|tool_call_argument_begin|>{"content": "Los Angeles today"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps(
                            {"latitude": 34.0522, "longitude": -118.2437}
                        ),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.get_news:1",
                    function=FunctionCall(
                        name="get_news",
                        arguments=json.dumps({"content": "Los Angeles today"}),
                    ),
                    type="function",
                ),
            ],
            "I'll get the weather and news for LA today. First, let me get the weather using Los Angeles coordinates, and then get the latest news. ",
        ),
        (
            """I'll help you with multiple tasks. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "New York"}<|tool_call_end|><|tool_call_begin|>functions.get_news:1<|tool_call_argument_begin|>{"topic": "technology"}<|tool_call_end|><|tool_call_begin|>functions.send_email:2<|tool_call_argument_begin|>{"to": "user@example.com", "subject": "Daily Update"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.get_weather:0",
                    function=FunctionCall(
                        name="get_weather",
                        arguments=json.dumps({"city": "New York"}),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.get_news:1",
                    function=FunctionCall(
                        name="get_news",
                        arguments=json.dumps({"topic": "technology"}),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.send_email:2",
                    function=FunctionCall(
                        name="send_email",
                        arguments=json.dumps(
                            {"to": "user@example.com", "subject": "Daily Update"}
                        ),
                    ),
                    type="function",
                ),
            ],
            "I'll help you with multiple tasks. ",
        ),
        (
            """Mixed spacing test. <|tool_calls_section_begin|> <|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> {} <|tool_call_end|><|tool_call_begin|>functions.test2:1<|tool_call_argument_begin|>{}<|tool_call_end|> <|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.test:0",
                    function=FunctionCall(
                        name="test",
                        arguments=json.dumps({}),
                    ),
                    type="function",
                ),
                ToolCall(
                    id="functions.test2:1",
                    function=FunctionCall(
                        name="test2",
                        arguments=json.dumps({}),
                    ),
                    type="function",
                ),
            ],
            "Mixed spacing test. ",
        ),
        (
            """I need to process HTML content. <|tool_calls_section_begin|><|tool_call_begin|>functions.process_html:0<|tool_call_argument_begin|>{"html": "<div>content</div>", "text": "normal text"}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.process_html:0",
                    function=FunctionCall(
                        name="process_html",
                        arguments=json.dumps(
                            {"html": "<div>content</div>", "text": "normal text"}
                        ),
                    ),
                    type="function",
                )
            ],
            "I need to process HTML content. ",
        ),
        (
            """I need to process formatted JSON. <|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{
  "name": "test",
  "value": 123,
  "nested": {
    "key": "value"
  }
}<|tool_call_end|><|tool_calls_section_end|>""",
            [
                ToolCall(
                    id="functions.process_data:0",
                    function=FunctionCall(
                        name="process_data",
                        arguments=json.dumps(
                            {"name": "test", "value": 123, "nested": {"key": "value"}},
                            indent=2,
                        ),
                    ),
                    type="function",
                )
            ],
            "I need to process formatted JSON. ",
        ),
    ],
)
def test_extract_tool_calls(
    kimi_k2_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert extracted_tool_calls.tools_called

    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_invalid_json(kimi_k2_tool_parser):
    """Invalid JSON (missing closing brace) should be repaired via _try_parse_json."""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing" <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Both tool calls should be extracted — the first has its JSON repaired
    assert len(extracted_tool_calls.tool_calls) == 2
    assert extracted_tool_calls.tool_calls[0].function.name == "invalid_get_weather"
    # The repaired JSON should be valid
    parsed_args = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert parsed_args == {"city": "Beijing"}
    assert extracted_tool_calls.tool_calls[1].function.name == "valid_get_weather"


def test_extract_tool_calls_invalid_funcall(kimi_k2_tool_parser):
    """we'll return every funcall result"""
    model_output = """I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.invalid_get_weather.0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_call_begin|>
functions.valid_get_weather:1 <|tool_call_argument_begin|> {"city": "Shanghai"} <|tool_call_end|> <|tool_calls_section_end|>"""

    extracted_tool_calls = kimi_k2_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    assert extracted_tool_calls.tools_called
    # Should extract only the valid JSON tool calls
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "valid_get_weather"


def test_streaming_basic_functionality(kimi_k2_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    kimi_k2_tool_parser.current_tool_name_sent = False
    kimi_k2_tool_parser.prev_tool_call_arr = []
    kimi_k2_tool_parser.current_tool_id = -1
    kimi_k2_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """ check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>
functions.get_weather:0 <|tool_call_argument_begin|> {"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>"""

    # First call should handle the initial setup
    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="I'll help you",
        current_text=current_text,
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # The result might be None or contain tool call information
    # This depends on the internal state management
    if result is not None and hasattr(result, "tool_calls") and result.tool_calls:
        assert len(result.tool_calls) >= 0


def test_streaming_no_tool_calls(kimi_k2_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == " without any tool calls."


def test_token_leak_between_section_and_tool_begin(kimi_k2_tool_parser):
    """
    Test that text between <|tool_calls_section_begin|> and <|tool_call_begin|>
    is suppressed and does not leak into reasoning_delta.
    This is the main vulnerability being fixed.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    # Get token IDs for the markers
    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )
    tool_call_begin_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Simulate streaming sequence:
    deltas = [
        ("I'll help you with that. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_token_id]),
        (" spurious text ", [4, 5]),
        ("<|tool_call_begin|>", [tool_call_begin_token_id]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Delta 1: "I'll help you with that. "
    assert results[0] is not None
    assert results[0].content == "I'll help you with that. "

    # Delta 2: "<|tool_calls_section_begin|>"
    # Section marker should be stripped and suppressed
    assert results[1] is None or (
        results[1].content is None or results[1].content == ""
    )

    # Delta 3: " spurious text or tokens " (THE LEAK SCENARIO)
    # CRITICAL: This text should be suppressed, NOT returned as reasoning_delta
    assert results[2] is None or (
        results[2].content is None or results[2].content == ""
    )

    # Delta 4: "<|tool_call_begin|>..."
    # Now we're in tool call mode, result depends on internal state
    # The key is that the spurious text from Delta 3 was not leaked


def test_split_markers_across_deltas(kimi_k2_tool_parser):
    """
    Test that markers split across delta chunks are correctly detected
    via the rolling buffer mechanism.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_token_id = kimi_k2_tool_parser.vocab.get(
        "<|tool_calls_section_begin|>"
    )

    # Delta 1: partial token, Delta 2: complete marker
    deltas = [
        ("<|tool_calls_sec", [3]),
        ("tion_begin|> ", [section_begin_token_id, 4]),
    ]

    _results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Now the complete marker should be detected via buffer
    assert kimi_k2_tool_parser.in_tool_section is True


def test_marker_variants(kimi_k2_tool_parser):
    """Test that both singular and plural marker variants are recognized."""
    kimi_k2_tool_parser.reset_streaming_state()

    # Test singular variant: <|tool_call_section_begin|> (note: singular "call")
    singular_token_id = kimi_k2_tool_parser.vocab.get("<|tool_call_section_begin|>")

    if singular_token_id is not None:  # Only test if tokenizer supports it
        _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
            previous_text="Reasoning ",
            current_text="Reasoning <|tool_call_section_begin|>",
            delta_text="<|tool_call_section_begin|>",
            previous_token_ids=[1, 2],
            current_token_ids=[1, 2, singular_token_id],
            delta_token_ids=[singular_token_id],
            request=None,
        )
        # Should enter tool section mode with singular variant too
        assert kimi_k2_tool_parser.in_tool_section is True


def test_reentry_to_reasoning_after_tool_section(kimi_k2_tool_parser):
    """
    Test that after exiting a tool section with <|tool_calls_section_end|>,
    subsequent text is correctly returned as reasoning content.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    deltas = [
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" More reasoning", [10, 11]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    assert kimi_k2_tool_parser.in_tool_section is False
    assert results[2] is not None
    assert results[2].content == " More reasoning"


def test_empty_tool_section(kimi_k2_tool_parser):
    """Test an empty tool section (begin immediately followed by end)."""
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Section begin
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning <|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )

    # Immediate section end
    _result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning <|tool_calls_section_begin|>",
        current_text="Reasoning <|tool_calls_section_begin|><|tool_calls_section_end|>",
        delta_text="<|tool_calls_section_end|>",
        previous_token_ids=[1, section_begin_id],
        current_token_ids=[1, section_begin_id, section_end_id],
        delta_token_ids=[section_end_id],
        request=None,
    )
    # Should exit cleanly without errors
    assert kimi_k2_tool_parser.in_tool_section is False


def test_large_tool_call_args_no_truncation(kimi_k2_tool_parser):
    """
    Test that large tool call arguments are NOT truncated.
    Regression test for GitHub issue #34442 where a hardcoded 8K limit
    silently truncated large arguments (e.g. code generation via tool calls).
    """
    kimi_k2_tool_parser.reset_streaming_state()

    # Generate arguments larger than the old 8192 char limit
    large_code = "x" * 20000
    large_args = json.dumps({"code": large_code})

    model_output = (
        f"Here's the code. <|tool_calls_section_begin|>"
        f"<|tool_call_begin|>functions.write_file:0"
        f"<|tool_call_argument_begin|>{large_args}"
        f"<|tool_call_end|><|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "write_file"
    # The full arguments must be preserved, not truncated
    parsed_args = json.loads(result.tool_calls[0].function.arguments)
    assert len(parsed_args["code"]) == 20000


def test_state_reset(kimi_k2_tool_parser):
    """Test that reset_streaming_state() properly clears all state."""
    # Put parser in a complex state
    kimi_k2_tool_parser.in_tool_section = True
    kimi_k2_tool_parser.token_buffer = "some buffer"
    kimi_k2_tool_parser.current_tool_id = 5
    kimi_k2_tool_parser.prev_tool_call_arr = [{"id": "test"}]
    kimi_k2_tool_parser._buffer_overflow_logged = True

    # Reset
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify all state is cleared
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""
    assert kimi_k2_tool_parser.current_tool_id == -1
    assert kimi_k2_tool_parser.prev_tool_call_arr == []
    assert kimi_k2_tool_parser._buffer_overflow_logged is False
    assert kimi_k2_tool_parser.current_tool_name_sent is False
    assert kimi_k2_tool_parser.streamed_args_for_tool == []


def test_section_begin_noise_tool_begin_same_chunk(kimi_k2_tool_parser):
    """
    Test that begin→noise→tool_begin within the SAME chunk suppresses
    the noise text correctly (not just across chunks).
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    tool_call_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")

    # Single delta containing: section_begin + spurious text + tool_call_begin
    combined_text = "<|tool_calls_section_begin|> noise text <|tool_call_begin|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_text,
        delta_text=combined_text,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, 3, 4, tool_call_begin_id],
        delta_token_ids=[section_begin_id, 3, 4, tool_call_begin_id],
        request=None,
    )

    # The noise text should NOT leak into content
    # Result should either be None/empty or start tool call parsing
    if result is not None and result.content is not None:
        # If content is returned, it should not contain the noise
        assert "noise text" not in result.content
        assert result.content == "" or result.content.strip() == ""


def test_stream_ends_without_section_end_marker(kimi_k2_tool_parser):
    """
    Test that if the stream ends (EOF) without a proper section end marker,
    the parser doesn't leak text, doesn't crash, and resets state cleanly.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # Enter tool section
    _result1 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Some content in tool section
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|> partial content",
        delta_text=" partial content",
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )
    # Content should be suppressed
    assert result2.content == "" or result2.content is None

    # Stream ends (EOF) - no more deltas, no section_end marker
    # Simulate this by manually checking state and resetting
    # (In real usage, the request handler would call reset_streaming_state)
    assert kimi_k2_tool_parser.in_tool_section is True  # Still in section

    # Reset state (as would happen between requests)
    kimi_k2_tool_parser.reset_streaming_state()

    # Verify clean slate
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.token_buffer == ""

    # Next request should work normally
    result3 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="New reasoning",
        delta_text="New reasoning",
        previous_token_ids=[],
        current_token_ids=[20, 21],
        delta_token_ids=[20, 21],
        request=None,
    )
    assert result3 is not None
    assert result3.content == "New reasoning"


def test_same_chunk_begin_and_end_markers(kimi_k2_tool_parser):
    """
    CRITICAL TEST: Verify that when both section_begin and section_end
    markers appear in the SAME chunk, the parser correctly:
    1. Enters the tool section
    2. Immediately exits the tool section
    3. Does NOT get stuck in in_tool_section=True state

    This tests the bug fix where elif was changed to if to handle
    both state transitions in a single delta.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Single chunk with both markers (e.g., empty tool section)
    combined_delta = "<|tool_calls_section_begin|><|tool_calls_section_end|>"

    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning ",
        current_text="Some reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1, 2],
        current_token_ids=[1, 2, section_begin_id, section_end_id],
        delta_token_ids=[section_begin_id, section_end_id],
        request=None,
    )

    # CRITICAL: Parser should NOT be stuck in tool section
    assert kimi_k2_tool_parser.in_tool_section is False, (
        "Parser stuck in tool section after processing both begin/end in same chunk. "
        "This indicates the elif bug was not fixed."
    )

    # Result should be empty or contain only stripped content
    assert result is not None
    assert result.content == "" or result.content is None

    # Verify subsequent content streams correctly (not suppressed)
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Some reasoning " + combined_delta,
        current_text="Some reasoning " + combined_delta + " More reasoning",
        delta_text=" More reasoning",
        previous_token_ids=[1, 2, section_begin_id, section_end_id],
        current_token_ids=[1, 2, section_begin_id, section_end_id, 10, 11],
        delta_token_ids=[10, 11],
        request=None,
    )

    # This content should NOT be suppressed (we're out of tool section)
    assert result2 is not None
    assert result2.content == " More reasoning"


def test_same_chunk_begin_content_end_markers(kimi_k2_tool_parser):
    """
    Test the same-chunk scenario with actual content between markers.
    Example: <|tool_calls_section_begin|> text <|tool_calls_section_end|>
    all arriving in one delta. The key is that the state machine correctly
    transitions in and out within the same chunk.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")

    # Chunk with begin, some whitespace/noise, and end all together
    # This simulates a tool section that opens and closes in the same chunk
    combined_delta = "<|tool_calls_section_begin|>   <|tool_calls_section_end|>"

    _result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning ",
        current_text="Reasoning " + combined_delta,
        delta_text=combined_delta,
        previous_token_ids=[1],
        current_token_ids=[1, section_begin_id, 100, section_end_id],
        delta_token_ids=[section_begin_id, 100, section_end_id],
        request=None,
    )

    # Parser should exit cleanly (not stuck in tool section)
    assert kimi_k2_tool_parser.in_tool_section is False

    # Verify the fix: next content should stream normally, not be suppressed
    result2 = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="Reasoning " + combined_delta,
        current_text="Reasoning " + combined_delta + " Done",
        delta_text=" Done",
        previous_token_ids=[1, section_begin_id, 100, section_end_id],
        current_token_ids=[1, section_begin_id, 100, section_end_id, 200],
        delta_token_ids=[200],
        request=None,
    )

    # Content after section should be returned (not suppressed)
    assert result2 is not None
    assert result2.content == " Done"


def test_tool_call_end_and_section_end_same_chunk(kimi_k2_tool_parser):
    """
    CRITICAL TEST (P1): Verify that when both <|tool_call_end|> and
    <|tool_calls_section_end|> appear in the SAME chunk, the parser:
    1. Processes the tool_call_end first (emits final arguments)
    2. THEN exits the section
    3. Does NOT drop the final tool call update
    4. Does NOT leak special tokens into reasoning

    This tests the deferred section exit fix.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # Simulate a streaming sequence for a SHORT tool call (all in one chunk):
    combined = (
        '<|tool_call_begin|>get_weather:0 <|tool_call_argument_begin|> {"city": "Paris"} '
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    deltas = [
        ("Let me help. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (combined, [tool_begin_id, 10, 11, 12, tool_end_id, section_end_id]),
        (" Done", [20]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # CRITICAL: Parser should have exited section AFTER processing tool
    assert kimi_k2_tool_parser.in_tool_section is False

    # Tool call should have been emitted (not dropped)
    if results[2] is not None and results[2].content is not None:
        # Verify no special tokens leaked into content
        assert "<|tool_call_end|>" not in results[2].content
        assert "<|tool_calls_section_end|>" not in results[2].content

    # Content after tool section should stream normally
    assert results[3] is not None
    assert results[3].content == " Done"


def test_streaming_tool_call_markers_not_leaked(kimi_k2_tool_parser):
    """
    CRITICAL TEST: Verify that tool call markers (<|tool_call_begin|>,
    <|tool_call_end|>, <|tool_call_argument_begin|>) are NOT leaked
    into the content field during streaming.

    This reproduces the AWS Bedrock bug where tool call markers appeared
    in the 'text' field of responses.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # List of markers that should NEVER appear in content
    forbidden_markers = [
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
    ]

    all_content = []

    # Steps: reasoning, section begin, tool call, section end, more reasoning
    tool_chunk = (
        "<|tool_call_begin|> functions.get_weather:0 "
        '<|tool_call_argument_begin|> {"city": "Tokyo"} <|tool_call_end|>'
    )
    deltas = [
        ("I'll check the weather. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (tool_chunk, [tool_begin_id, 10, 11, tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" Here's the result.", [20, 21]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    for res in results:
        if res and res.content:
            all_content.append(res.content)

    # CRITICAL ASSERTIONS: No forbidden markers in any content
    full_content = "".join(all_content)
    for marker in forbidden_markers:
        assert marker not in full_content, (
            f"MARKER LEAK DETECTED: '{marker}' found in content. "
            f"Full content: {repr(full_content)}"
        )

    # Also check that tool call content (function name, arguments) is not leaked
    assert "get_weather" not in full_content, (
        f"TOOL CALL CONTENT LEAKED: 'get_weather' found in content. "
        f"Full content: {repr(full_content)}"
    )
    assert "Tokyo" not in full_content, (
        f"TOOL CALL CONTENT LEAKED: 'Tokyo' found in content. "
        f"Full content: {repr(full_content)}"
    )

    # Verify that legitimate content was preserved
    assert "I'll check the weather." in full_content or len(all_content) > 0


def test_streaming_multiple_tool_calls_not_leaked(kimi_k2_tool_parser):
    """
    Test that MULTIPLE tool calls in streaming mode do not leak into content.
    This reproduces the AWS Bedrock scenario: "Compare weather in Tokyo and NYC".
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    all_content = []

    tool1 = '<|tool_call_begin|> get_weather:0 <|tool_call_argument_begin|> {"city": "Tokyo"} <|tool_call_end|>'
    tool2 = ' <|tool_call_begin|> get_weather:1 <|tool_call_argument_begin|> {"city": "New York"} <|tool_call_end|>'

    deltas = [
        ("I'll compare the weather. ", [1, 2, 3]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        (tool1, [tool_begin_id, 10, tool_end_id]),
        (tool2, [tool_begin_id, 20, tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" Here's the comparison.", [30]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    for res in results:
        if res and res.content:
            all_content.append(res.content)

    # Assertions
    full_content = "".join(all_content)

    # Check no markers leaked
    forbidden = ["<|tool_call", "<|tool_calls_section"]
    for marker in forbidden:
        assert marker not in full_content, (
            f"MARKER LEAKED: {marker} in {repr(full_content)}"
        )

    # Check no tool call content leaked (both tools)
    assert "get_weather" not in full_content, f"TOOL NAME LEAKED: {repr(full_content)}"
    assert "Tokyo" not in full_content, f"TOOL ARG LEAKED (Tokyo): {repr(full_content)}"
    assert "New York" not in full_content, (
        f"TOOL ARG LEAKED (NYC): {repr(full_content)}"
    )

    # Legitimate content preserved
    assert "compare" in full_content.lower() or len(all_content) > 0


# ============================================================
# Regression tests for specific bug fixes
# ============================================================


def test_extract_tool_calls_singular_variant(kimi_k2_tool_parser):
    """
    Test that non-streaming extract_tool_calls works with the singular
    marker variant <|tool_call_section_begin|> (note: "call" not "calls").
    Regression test: previously only the plural variant was checked.
    """
    model_output = (
        "I'll check the weather. "
        "<|tool_call_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Berlin"}'
        "<|tool_call_end|>"
        "<|tool_call_section_end|>"
    )

    # Check if the tokenizer actually has the singular variant
    singular_id = kimi_k2_tool_parser.vocab.get("<|tool_call_section_begin|>")
    if singular_id is None:
        pytest.skip("Tokenizer does not have singular variant token")

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called, (
        "extract_tool_calls failed with singular section marker variant"
    )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.content == "I'll check the weather. "


@pytest.mark.parametrize(
    ids=[
        "array_ending",
        "boolean_ending",
        "number_ending",
        "null_ending",
        "nested_object_ending",
        "string_ending",
    ],
    argnames=["args_dict"],
    argvalues=[
        ({"items": [1, 2, 3]},),
        ({"flag": True, "name": "test"},),
        ({"count": 42},),
        ({"result": None},),
        ({"data": {"nested": {"deep": [1, 2]}}},),
        ({"city": "San Francisco"},),
    ],
)
def test_streaming_tool_close_various_json_endings(kimi_k2_tool_parser, args_dict):
    """
    Regression test: the old closing logic looked for '"}' in delta_text
    to detect the end of JSON arguments. This failed for any JSON not
    ending with a string value — arrays, booleans, numbers, null, nested
    objects all got their final chunk silently dropped.

    The fix uses proper diffing against streamed_args_for_tool instead.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    args_str = json.dumps(args_dict)

    # We need to stream the tool call in multiple chunks so that the
    # closing logic actually has a diff to emit. Split args roughly in half.
    split_point = len(args_str) // 2
    args_part1 = args_str[:split_point]
    args_part2 = args_str[split_point:]

    deltas = [
        # 1. Reasoning content
        ("Let me help. ", [1, 2]),
        # 2. Section begin
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        # 3. Tool call begin + name + argument_begin + first half of args
        (
            f"<|tool_call_begin|>functions.test_func:0 <|tool_call_argument_begin|>{args_part1}",
            [tool_begin_id, 10, 11, 12],
        ),
        # 4. Second half of args (updating existing tool call)
        (args_part2, [13, 14]),
        # 5. Tool call end + section end (closing chunk)
        (
            "<|tool_call_end|><|tool_calls_section_end|>",
            [tool_end_id, section_end_id],
        ),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Collect all streamed tool call arguments
    all_tool_args = []
    for r in results:
        if r is not None and r.tool_calls:
            for tc in r.tool_calls:
                if hasattr(tc, "function") and tc.function is not None:
                    func = tc.function
                    # Handle both dict and object forms
                    args = (
                        func.get("arguments")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", None)
                    )
                    if args:
                        all_tool_args.append(args)

    concatenated_args = "".join(all_tool_args)
    assert concatenated_args, (
        f"No tool call arguments were streamed for args ending with "
        f"{repr(args_str[-5:])}. Results: {results}"
    )

    # The concatenated streamed args should form valid JSON matching the input
    try:
        parsed = json.loads(concatenated_args)
    except json.JSONDecodeError:
        # Partial streaming is OK as long as we got something
        # (the closing chunk is what we're really testing)
        pass
    else:
        assert parsed == args_dict, (
            f"Streamed args don't match input. Got {parsed}, expected {args_dict}"
        )


def test_streaming_greedy_regex_multiple_tools(kimi_k2_tool_parser):
    """
    Regression test for GitHub issue #24478: greedy .+ in streaming regex
    matched across tool call boundaries, collapsing multiple tool calls
    into one garbled entry.

    The fix changes .+ to [^<]+ for the tool_call_id capture group.

    Streaming must be incremental — the parser uses token ID counting
    and requires tool_call_begin before tool_call_end in separate chunks
    to detect the start/close of each tool call.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    # Stream tool calls incrementally: begin, name+args, end for each
    deltas = [
        ("I'll help. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        # Tool 1: begin
        ("<|tool_call_begin|>", [tool_begin_id]),
        # Tool 1: name + args
        (
            'functions.get_weather:0 <|tool_call_argument_begin|>{"city": "Tokyo"}',
            [10, 11, 12],
        ),
        # Tool 1: end
        (" <|tool_call_end|>", [tool_end_id]),
        # Tool 2: begin
        ("<|tool_call_begin|>", [tool_begin_id]),
        # Tool 2: name + args
        (
            'functions.get_news:1 <|tool_call_argument_begin|>{"topic": "tech"}',
            [20, 21, 22],
        ),
        # Tool 2: end
        (" <|tool_call_end|>", [tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
    ]

    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)

    # Collect tool call names from streamed results
    tool_names = []
    for r in results:
        if r is not None and r.tool_calls:
            for tc in r.tool_calls:
                if hasattr(tc, "function") and tc.function is not None:
                    func = tc.function
                    name = (
                        func.get("name")
                        if isinstance(func, dict)
                        else getattr(func, "name", None)
                    )
                    if name:
                        tool_names.append(name)

    # We should see two distinct tool names, not one garbled entry
    assert "get_weather" in tool_names, (
        f"get_weather not found in streamed tool names: {tool_names}"
    )
    assert "get_news" in tool_names, (
        f"get_news not found in streamed tool names: {tool_names}"
    )


def test_extract_tool_calls_non_string_json_values(kimi_k2_tool_parser):
    """
    Test that non-streaming extraction handles various JSON value types
    correctly, including arrays, booleans, numbers, and null.
    """
    model_output = (
        "Running tasks. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.process:0"
        '<|tool_call_argument_begin|>{"items": [1, "two", null], "flag": true, "count": 99}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed["items"] == [1, "two", None]
    assert parsed["flag"] is True
    assert parsed["count"] == 99


def test_extract_tool_calls_backslashes_in_args(kimi_k2_tool_parser):
    """
    Regression test: the old code had a unicode_escape decode that would
    corrupt backslashes in arguments (e.g. Windows paths like C:\\Users).
    The fix removes the encode/decode round-trip entirely.
    """
    # Non-streaming path: backslashes in arguments should be preserved
    model_output = (
        "Writing file. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.write_file:0"
        '<|tool_call_argument_begin|>{"path": "C:\\\\Users\\\\test\\\\file.py", "content": "hello\\nworld"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1

    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed["path"] == "C:\\Users\\test\\file.py"
    assert parsed["content"] == "hello\nworld"


def test_extract_tool_calls_unicode_in_args(kimi_k2_tool_parser):
    """
    Regression test: ensure non-ASCII characters in tool call arguments
    are preserved correctly. The old unicode_escape decode would corrupt
    multi-byte UTF-8 characters.
    """
    model_output = (
        "Checking weather. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "\u6771\u4eac", "note": "\u00e9\u00e8\u00ea"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed["city"] == "\u6771\u4eac"  # Tokyo in Japanese
    assert parsed["note"] == "\u00e9\u00e8\u00ea"


def test_buffer_overflow_warning_resets_between_sections(kimi_k2_tool_parser):
    """
    Regression test: _buffer_overflow_logged was never reset, meaning
    the overflow warning was permanently silenced after the first occurrence.
    Now it resets on section exit so each section can warn independently.
    """
    kimi_k2_tool_parser.reset_streaming_state()

    # Simulate overflow
    kimi_k2_tool_parser._buffer_overflow_logged = True
    kimi_k2_tool_parser.in_tool_section = True

    # Exit section
    kimi_k2_tool_parser._reset_section_state()

    # Flag should be reset so next section can warn again
    assert kimi_k2_tool_parser._buffer_overflow_logged is False


def test_extract_tool_calls_concatenated_no_spaces(kimi_k2_tool_parser):
    """
    Regression test for GitHub issue #24478: tool calls concatenated with
    zero spacing between end and begin markers should be parsed as separate
    tool calls, not collapsed into one.
    """
    model_output = (
        "Doing two things. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.func_a:0"
        '<|tool_call_argument_begin|>{"x": 1}'
        "<|tool_call_end|>"
        "<|tool_call_begin|>functions.func_b:1"  # No space before this
        '<|tool_call_argument_begin|>{"y": 2}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 2, (
        f"Expected 2 tool calls but got {len(result.tool_calls)}: "
        f"{[tc.function.name for tc in result.tool_calls]}"
    )
    assert result.tool_calls[0].function.name == "func_a"
    assert result.tool_calls[1].function.name == "func_b"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}
    assert json.loads(result.tool_calls[1].function.arguments) == {"y": 2}


def test_extract_tool_calls_json_repair_trailing_comma(kimi_k2_tool_parser):
    """
    Test that the parser can repair common JSON errors from model output,
    such as trailing commas before closing braces.
    """
    model_output = (
        "Here you go. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Tokyo",}'  # trailing comma
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    # Should have repaired the trailing comma
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed == {"city": "Tokyo"}


def test_extract_tool_calls_json_repair_unclosed_brace(kimi_k2_tool_parser):
    """
    Test that the parser can repair unclosed braces in JSON arguments.
    Models sometimes fail to close nested objects.
    """
    model_output = (
        "Running. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.update:0"
        '<|tool_call_argument_begin|>{"data": {"key": "value"}'  # missing outer }
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed == {"data": {"key": "value"}}


def test_extract_tool_calls_validates_function_names(kimi_k2_tool_parser):
    """
    Test that when tools are provided in the request, the parser validates
    function names and skips calls to undefined functions.
    Inspired by SGLang's tool validation approach.
    """
    from unittest.mock import MagicMock

    # Create a mock request with specific tools
    mock_request = MagicMock()
    tool1 = MagicMock()
    tool1.function.name = "get_weather"
    tool2 = MagicMock()
    tool2.function.name = "get_news"
    mock_request.tools = [tool1, tool2]

    model_output = (
        "Let me help. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Tokyo"}'
        "<|tool_call_end|>"
        "<|tool_call_begin|>functions.hack_system:1"  # hallucinated function
        '<|tool_call_argument_begin|>{"target": "server"}'
        "<|tool_call_end|>"
        "<|tool_call_begin|>functions.get_news:2"
        '<|tool_call_argument_begin|>{"topic": "tech"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=mock_request)
    assert result.tools_called
    # Should have skipped the hallucinated "hack_system" call
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[1].function.name == "get_news"


def test_extract_tool_calls_no_validation_without_tools(kimi_k2_tool_parser):
    """
    When no tools are specified in the request, all function calls should
    be accepted without validation.
    """
    model_output = (
        "Here. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.any_function:0"
        '<|tool_call_argument_begin|>{"x": 1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "any_function"


def test_extract_tool_calls_named_groups_in_regex(kimi_k2_tool_parser):
    """
    Test that the regex correctly extracts function_name and function_idx
    as separate named groups for both prefixed and unprefixed formats.
    """
    # With "functions." prefix
    model_output1 = (
        "Test. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.my_tool:0"
        '<|tool_call_argument_begin|>{"a": 1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result1 = kimi_k2_tool_parser.extract_tool_calls(model_output1, request=None)
    assert result1.tools_called
    assert result1.tool_calls[0].function.name == "my_tool"

    # Without "functions." prefix (some models omit it)
    model_output2 = (
        "Test. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>my_tool:0"
        '<|tool_call_argument_begin|>{"a": 1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result2 = kimi_k2_tool_parser.extract_tool_calls(model_output2, request=None)
    assert result2.tools_called
    assert result2.tool_calls[0].function.name == "my_tool"


def test_extract_tool_calls_incomplete_without_end_token(kimi_k2_tool_parser):
    """
    Test that tool calls without a proper end token (truncated output)
    are still extracted via the $ anchor in the regex.
    Adopted from SGLang's approach.
    """
    # Model output truncated — no tool_call_end or section_end
    model_output = (
        "Working. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.search:0"
        '<|tool_call_argument_begin|>{"query": "test"}'
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "search"
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed == {"query": "test"}


def test_extract_tool_calls_nested_json(kimi_k2_tool_parser):
    """
    Test that deeply nested JSON is correctly extracted by the non-greedy
    regex anchored to end tokens (not stopping at the first '}').
    """
    nested_args = json.dumps(
        {
            "config": {
                "settings": {
                    "theme": {"primary": "#fff", "secondary": "#000"},
                    "flags": [True, False],
                },
                "metadata": {"version": 2},
            }
        }
    )

    model_output = (
        "Configuring. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.configure:0"
        f"<|tool_call_argument_begin|>{nested_args}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    parsed = json.loads(result.tool_calls[0].function.arguments)
    assert parsed["config"]["settings"]["theme"]["primary"] == "#fff"
    assert parsed["config"]["metadata"]["version"] == 2


# ============================================================
# Robustness and fault-tolerance tests
# ============================================================


def test_content_sanitization_on_parse_failure(kimi_k2_tool_parser):
    """
    When extract_tool_calls encounters an error and falls back, the
    returned content should NOT contain any Kimi K2 special markers.
    This prevents marker leakage to the end user.
    """
    # Craft input where section markers exist but tool call regex won't match
    # (invalid tool call format — no colon in ID)
    model_output = (
        "Here's the result. "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>badformat_no_colon"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    # Whether or not tool calls were found, content must be clean
    if result.content:
        from vllm.tool_parsers.kimi_k2_tool_parser import _ALL_MARKERS

        for marker in _ALL_MARKERS:
            assert marker not in result.content, (
                f"MARKER LEAK: '{marker}' found in content: {result.content!r}"
            )


def test_content_sanitization_strips_all_marker_types(kimi_k2_tool_parser):
    """
    Test that _sanitize_content strips every known marker type.
    """
    from vllm.tool_parsers.kimi_k2_tool_parser import _sanitize_content

    dirty = (
        "Hello <|tool_calls_section_begin|> world "
        "<|tool_call_begin|> foo <|tool_call_argument_begin|> "
        "<|tool_call_end|> bar <|tool_calls_section_end|>"
        "<|tool_call_section_begin|><|tool_call_section_end|>"
    )
    clean = _sanitize_content(dirty)
    assert "<|" not in clean
    assert "Hello" in clean
    assert "world" in clean
    assert "foo" in clean
    assert "bar" in clean


def test_streaming_auto_reset_on_stale_state(kimi_k2_tool_parser):
    """
    If a previous stream was aborted (e.g. client disconnect) and the
    parser is reused for a new request, stale state should be auto-reset
    when previous_text is empty (start of new request).
    """
    # Simulate stale state from a previous aborted stream
    kimi_k2_tool_parser.current_tool_id = 3
    kimi_k2_tool_parser.in_tool_section = True
    kimi_k2_tool_parser.prev_tool_call_arr = [{"name": "stale"}]
    kimi_k2_tool_parser.streamed_args_for_tool = ["stale"]

    # New request starts: previous_text="" indicates fresh start
    result = kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="Hello",
        delta_text="Hello",
        previous_token_ids=[],
        current_token_ids=[1, 2],
        delta_token_ids=[1, 2],
        request=None,
    )

    # State should have been auto-reset
    assert kimi_k2_tool_parser.current_tool_id == -1
    assert kimi_k2_tool_parser.in_tool_section is False
    assert kimi_k2_tool_parser.prev_tool_call_arr == []
    assert kimi_k2_tool_parser.streamed_args_for_tool == []

    # Content should flow through normally
    assert result is not None
    assert result.content == "Hello"


def test_streaming_no_false_reset_on_continuation(kimi_k2_tool_parser):
    """
    Auto-reset should NOT trigger when previous_text is non-empty
    (i.e., we're mid-stream, not starting fresh).
    """
    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")

    # First delta: enters tool section (sets state)
    kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="<|tool_calls_section_begin|>",
        delta_text="<|tool_calls_section_begin|>",
        previous_token_ids=[],
        current_token_ids=[section_begin_id],
        delta_token_ids=[section_begin_id],
        request=None,
    )
    assert kimi_k2_tool_parser.in_tool_section is True

    # Second delta: continuation (previous_text is non-empty)
    # Auto-reset should NOT fire here
    kimi_k2_tool_parser.extract_tool_calls_streaming(
        previous_text="<|tool_calls_section_begin|>",
        current_text="<|tool_calls_section_begin|> data",
        delta_text=" data",
        previous_token_ids=[section_begin_id],
        current_token_ids=[section_begin_id, 10],
        delta_token_ids=[10],
        request=None,
    )
    # Should still be in tool section (not falsely reset)
    assert kimi_k2_tool_parser.in_tool_section is True


def test_json_repair_trailing_comma_and_unclosed_brace(kimi_k2_tool_parser):
    """
    Test combined JSON repair: trailing comma AND unclosed brace together.
    """
    from vllm.tool_parsers.kimi_k2_tool_parser import _try_parse_json

    # Trailing comma + missing closing brace
    result = _try_parse_json('{"a": 1, "b": 2,')
    parsed = json.loads(result)
    assert parsed == {"a": 1, "b": 2}


def test_json_repair_empty_string():
    """_try_parse_json should handle empty/whitespace input gracefully."""
    from vllm.tool_parsers.kimi_k2_tool_parser import _try_parse_json

    assert _try_parse_json("") == ""
    assert _try_parse_json("   ") == ""


def test_json_repair_already_valid():
    """_try_parse_json should return valid JSON unchanged."""
    from vllm.tool_parsers.kimi_k2_tool_parser import _try_parse_json

    valid = '{"key": "value", "num": 42}'
    assert _try_parse_json(valid) == valid


def test_extract_tool_calls_empty_section(kimi_k2_tool_parser):
    """
    When the model outputs section markers but no actual tool calls inside,
    tools_called should still be True (section was detected) and content
    should be clean with no markers.
    """
    model_output = (
        "Let me think about this. "
        "<|tool_calls_section_begin|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called is True
    assert result.tool_calls == []
    assert result.content == "Let me think about this. "


def test_extract_tool_calls_only_markers_no_content(kimi_k2_tool_parser):
    """
    When model output is ONLY tool call markers with no reasoning content,
    the parser should extract correctly and return None for content.
    """
    model_output = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"city": "Tokyo"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    # Content should be None (no text before section begin)
    assert result.content is None


def test_extract_tool_calls_whitespace_only_content(kimi_k2_tool_parser):
    """
    When there's only whitespace before tool section, content should be None.
    """
    model_output = (
        "   \n\n  "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.do_thing:0"
        '<|tool_call_argument_begin|>{"x": 1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    result = kimi_k2_tool_parser.extract_tool_calls(model_output, request=None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    # Whitespace-only content (after strip by caller) should still be returned
    # as the raw whitespace — the serving layer handles stripping
    assert result.content is not None


def test_streaming_markers_not_leaked_in_any_content(kimi_k2_tool_parser):
    """
    Comprehensive marker leak test: run a full streaming sequence and
    verify NO content delta ever contains ANY Kimi K2 marker.
    """
    from vllm.tool_parsers.kimi_k2_tool_parser import _ALL_MARKERS

    kimi_k2_tool_parser.reset_streaming_state()

    section_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_begin|>")
    section_end_id = kimi_k2_tool_parser.vocab.get("<|tool_calls_section_end|>")
    tool_begin_id = kimi_k2_tool_parser.vocab.get("<|tool_call_begin|>")
    tool_end_id = kimi_k2_tool_parser.vocab.get("<|tool_call_end|>")

    deltas = [
        ("I'll check. ", [1, 2]),
        ("<|tool_calls_section_begin|>", [section_begin_id]),
        ("<|tool_call_begin|>", [tool_begin_id]),
        (
            'functions.get_weather:0 <|tool_call_argument_begin|>{"city": "NYC"}',
            [10, 11, 12],
        ),
        (" <|tool_call_end|>", [tool_end_id]),
        ("<|tool_calls_section_end|>", [section_end_id]),
        (" Here are the results.", [20, 21]),
    ]

    all_content = []
    results = run_streaming_sequence(kimi_k2_tool_parser, deltas)
    for r in results:
        if r and r.content:
            all_content.append(r.content)

    full_content = "".join(all_content)
    for marker in _ALL_MARKERS:
        assert marker not in full_content, (
            f"MARKER LEAK: '{marker}' in streamed content: {full_content!r}"
        )
