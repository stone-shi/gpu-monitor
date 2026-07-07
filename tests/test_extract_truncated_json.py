import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_llm_monitor import extract_truncated_json_field


class TestExtractTruncatedJsonField:
    def test_truncated_simple(self):
        line = '{"type":"llm.prediction.output","output":"hello world'
        result = extract_truncated_json_field(line, "output")
        assert result == "hello world"

    def test_truncated_with_space_after_colon(self):
        line = '{"type": "test", "output": "hello world'
        result = extract_truncated_json_field(line, "output")
        assert result == "hello world"

    def test_missing_field(self):
        line = '{"type":"test","data":"something'
        result = extract_truncated_json_field(line, "output")
        assert result is None

    def test_truncated_with_trailing_odd_backslash(self):
        line = '{"output":"hello\\'
        result = extract_truncated_json_field(line, "output")
        assert result == "hello"

    def test_truncated_with_trailing_even_backslashes(self):
        line = '{"output":"hello\\\\'
        result = extract_truncated_json_field(line, "output")
        assert result == "hello\\"

    def test_escaped_newlines_fallback(self):
        line = '{"output":"line1\\nline2'
        result = extract_truncated_json_field(line, "output")
        assert result == "line1\nline2"

    def test_escaped_tabs_fallback(self):
        line = '{"output":"col1\\tcol2'
        result = extract_truncated_json_field(line, "output")
        assert result == "col1\tcol2"

    def test_escaped_quotes_fallback(self):
        line = '{"output":"say \\"hi\\"'
        result = extract_truncated_json_field(line, "output")
        assert result == 'say "hi"'

    def test_empty_truncated(self):
        line = '{"output":'
        result = extract_truncated_json_field(line, "output")
        assert result is None

    def test_input_field_truncated(self):
        line = '{"type":"llm.prediction.input","input":"user prompt here'
        result = extract_truncated_json_field(line, "input")
        assert result == "user prompt here"

    def test_complete_value_with_closing_quote_and_brace(self):
        line = '{"output":"hello world"}'
        result = extract_truncated_json_field(line, "output")
        assert result == 'hello world"}'

    def test_trailing_newline_stripped(self):
        line = '{"output":"hello'
        result = extract_truncated_json_field(line, "output")
        assert result == "hello"

    def test_multiline_content(self):
        line = '{"output":"line1\\nline2\\nline3'
        result = extract_truncated_json_field(line, "output")
        assert result == "line1\nline2\nline3"

    def test_no_space_marker_preferred(self):
        line = '{"output":"value'
        result = extract_truncated_json_field(line, "output")
        assert result == "value"
