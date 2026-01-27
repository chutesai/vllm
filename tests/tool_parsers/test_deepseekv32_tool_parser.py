# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser

MODEL = "deepseek-ai/DeepSeek-V3"


@pytest.fixture(scope="module")
def deepseekv32_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(deepseekv32_tokenizer):
    return DeepSeekV32ToolParser(deepseekv32_tokenizer)


# =============================================================================
# Standard DSML format tests
# =============================================================================


def test_extract_tool_calls_standard_dsml(parser):
    """Test standard DSML format with full markers."""
    model_output = (
        "Here's the weather:\n"
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Tokyo</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}
    # Content preserves whitespace (newline before tool call)
    assert result.content == "Here's the weather:\n"


def test_extract_tool_calls_multiple_tools(parser):
    """Test multiple tool calls in standard format."""
    model_output = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Tokyo</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}
    assert result.tool_calls[1].function.name == "get_weather"
    assert json.loads(result.tool_calls[1].function.arguments) == {"location": "Paris"}


# =============================================================================
# Flexible format tests (without DSML markers)
# =============================================================================


def test_extract_tool_calls_no_dsml_markers(parser):
    """Test format without DSML markers - just plain XML tags."""
    model_output = (
        "Let me check:\n"
        "<function_calls>\n"
        '<invoke name="search">\n'
        '<parameter name="query" string="true">python tutorial</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "search"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "query": "python tutorial"
    }
    assert result.content == "Let me check:\n"


def test_extract_tool_calls_case_insensitive(parser):
    """Test case-insensitive matching."""
    model_output = (
        "<FUNCTION_CALLS>\n"
        '<INVOKE name="test_func">\n'
        '<PARAMETER name="arg" string="true">value</PARAMETER>\n'
        "</INVOKE>\n"
        "</FUNCTION_CALLS>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "test_func"


# =============================================================================
# Direct JSON parameter tests
# =============================================================================


def test_extract_tool_calls_json_params(parser):
    """Test direct JSON parameters inside invoke block."""
    model_output = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="create_user">\n'
        '{"name": "Alice", "age": 30, "active": true}\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "create_user"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"name": "Alice", "age": 30, "active": True}


def test_extract_tool_calls_json_params_no_dsml(parser):
    """Test direct JSON parameters without DSML markers."""
    model_output = (
        "<function_calls>\n"
        '<invoke name="calculate">\n'
        '{"x": 10, "y": 20, "operation": "add"}\n'
        "</invoke>\n"
        "</function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "calculate"
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"x": 10, "y": 20, "operation": "add"}


# =============================================================================
# Optional string attribute tests
# =============================================================================


def test_extract_tool_calls_no_string_attr(parser):
    """Test parameter tags without the string attribute."""
    model_output = (
        "<function_calls>\n"
        '<invoke name="greet">\n'
        '<parameter name="message">Hello World</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "greet"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "message": "Hello World"
    }


# =============================================================================
# Edge cases
# =============================================================================


def test_extract_tool_calls_no_tool_call(parser):
    """Test output with no tool calls."""
    model_output = "This is just regular text with no tool calls."
    result = parser.extract_tool_calls(model_output, None)
    assert not result.tools_called
    assert len(result.tool_calls) == 0
    assert result.content == model_output


def test_extract_tool_calls_empty_params(parser):
    """Test tool call with no parameters."""
    model_output = (
        "<function_calls>\n"
        '<invoke name="get_time">\n'
        "</invoke>\n"
        "</function_calls>"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_time"
    assert json.loads(result.tool_calls[0].function.arguments) == {}


def test_extract_tool_calls_whitespace_variations(parser):
    """Test with various whitespace in tags."""
    model_output = (
        "<  function_calls  >\n"
        '<  invoke   name = "test"  >\n'
        '<  parameter   name = "x"   string = "true"  >val</  parameter  >\n'
        "</  invoke  >\n"
        "</  function_calls  >"
    )
    result = parser.extract_tool_calls(model_output, None)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "test"
