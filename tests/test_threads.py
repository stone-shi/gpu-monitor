import os
import sys
import time
import json
import base64
import collections
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_llm_monitor import (
    hardware_polling_thread,
    localai_trace_polling_thread,
    process_trace,
    parse_streaming_response,
    parse_json_response,
    extract_last_user_message,
    parse_trace_timestamp,
)


class TestHardwarePollingThread:
    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.pynvml")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_nvml_init_failure_retries(self, mock_db, mock_pynvml, mock_sleep):
        mock_pynvml.nvmlInit.side_effect = [Exception("no gpu"), None]
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = MagicMock()
        mock_mem.used = 1024 * 1024 * 512
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        call_count = [0]
        original_sleep = mock_sleep.side_effect

        def sleep_side_effect(seconds):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_side_effect

        with pytest.raises(KeyboardInterrupt):
            hardware_polling_thread()

        assert mock_pynvml.nvmlInit.call_count == 2

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.pynvml")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_gpu_handle_failure_exits(self, mock_db, mock_pynvml, mock_sleep):
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("no device")

        hardware_polling_thread()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.pynvml")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_successful_metrics_insert(self, mock_db, mock_pynvml, mock_sleep):
        mock_pynvml.nvmlInit.return_value = None
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = MagicMock()
        mock_mem.used = 1024 * 1024 * 2048
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 70

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        def sleep_then_raise(s):
            raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_then_raise

        with pytest.raises(KeyboardInterrupt):
            hardware_polling_thread()

        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args
        assert args[0][1] == (75, 2048, 70)
        mock_conn.commit.assert_called_once()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.pynvml")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_db_connection_failure_retries(self, mock_db, mock_pynvml, mock_sleep):
        mock_pynvml.nvmlInit.return_value = None
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_db.side_effect = [Exception("conn failed"), MagicMock(closed=False)]

        call_count = [0]

        def sleep_side_effect(s):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_side_effect

        with pytest.raises(KeyboardInterrupt):
            hardware_polling_thread()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.pynvml")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_vram_bytes_to_mb_conversion(self, mock_db, mock_pynvml, mock_sleep):
        mock_pynvml.nvmlInit.return_value = None
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 10
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = MagicMock()
        mock_mem.used = 4 * 1024 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 50

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            hardware_polling_thread()

        args = mock_cursor.execute.call_args
        assert args[0][1][1] == 4096


def _make_trace(req_body_obj, resp_body_text, path="/v1/chat/completions", status=200,
                 timestamp="2026-07-23T02:23:43.048034614Z", duration=2_000_000_000):
    return {
        "timestamp": timestamp,
        "duration": duration,
        "request": {
            "method": "POST",
            "path": path,
            "body": base64.b64encode(json.dumps(req_body_obj).encode()).decode(),
        },
        "response": {
            "status": status,
            "body": base64.b64encode(resp_body_text.encode()).decode(),
        },
    }


class TestExtractLastUserMessage:
    def test_plain_string_content(self):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
        assert extract_last_user_message(messages) == "hello"

    def test_picks_most_recent_user_message(self):
        messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "second"}]
        assert extract_last_user_message(messages) == "second"

    def test_multimodal_content_parts(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "image_url", "image_url": {}}]}]
        assert extract_last_user_message(messages) == "part one"

    def test_no_user_message_returns_none(self):
        assert extract_last_user_message([{"role": "system", "content": "sys"}]) is None

    def test_non_list_input_returns_none(self):
        assert extract_last_user_message(None) is None


class TestParseTraceTimestamp:
    def test_truncates_nanoseconds_to_microseconds(self):
        result = parse_trace_timestamp("2026-07-23T02:23:43.048034614Z")
        assert result is not None
        assert result.microsecond == 48034

    def test_none_input(self):
        assert parse_trace_timestamp(None) is None

    def test_invalid_format(self):
        assert parse_trace_timestamp("not-a-timestamp") is None


class TestParseJsonResponse:
    def test_extracts_fields(self):
        body = json.dumps({
            "id": "abc-123",
            "model": "qwen3.6-35b-a3b",
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        })
        trace_id, model_name, response_text, usage = parse_json_response(body)
        assert trace_id == "abc-123"
        assert model_name == "qwen3.6-35b-a3b"
        assert response_text == "hi there"
        assert usage == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}

    def test_invalid_json_returns_none_tuple(self):
        assert parse_json_response("not json") == (None, None, None, None)


class TestParseStreamingResponse:
    def test_reassembles_content_and_usage_from_chunks(self):
        chunks = [
            {"id": "s1", "model": "qwen", "choices": [{"delta": {"content": "Hel"}}]},
            {"id": "s1", "model": "qwen", "choices": [{"delta": {"content": "lo"}}]},
            {"id": "s1", "model": "qwen", "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        trace_id, model_name, response_text, usage = parse_streaming_response(body)
        assert trace_id == "s1"
        assert model_name == "qwen"
        assert response_text == "Hello"
        assert usage == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


class TestProcessTrace:
    def test_non_chat_path_ignored(self):
        trace = _make_trace({}, "{}", path="/v1/models")
        assert process_trace(trace) is None

    def test_non_200_status_ignored(self):
        trace = _make_trace({}, "{}", status=500)
        assert process_trace(trace) is None

    def test_missing_usage_falls_back_to_length_estimate(self):
        req = {"model": "qwen", "messages": [{"role": "user", "content": "a" * 40}]}
        resp = json.dumps({
            "id": "abc",
            "model": "qwen",
            "choices": [{"message": {"role": "assistant", "content": "b" * 20}}],
        })
        record = process_trace(_make_trace(req, resp))
        assert record["prompt_tokens"] == 10
        assert record["completion_tokens"] == 5
        assert record["total_tokens"] == 15

    def test_tokens_per_sec_from_duration(self):
        req = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
        resp = json.dumps({
            "id": "abc",
            "model": "qwen",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11},
        })
        record = process_trace(_make_trace(req, resp, duration=5_000_000_000))
        assert record["tokens_per_sec"] == 2.0


class TestLocalAITracePollingThread:
    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.fetch_traces")
    def test_missing_traces_url_exits_immediately(self, mock_fetch, mock_sleep):
        with patch.dict(os.environ, {}, clear=True):
            localai_trace_polling_thread()
        mock_fetch.assert_not_called()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.get_db_connection")
    @patch("gpu_llm_monitor.fetch_traces")
    @patch.dict(os.environ, {"LOCALAI_TRACES_URL": "http://localai:4012/api/traces"})
    def test_new_trace_inserted_after_bootstrap(self, mock_fetch, mock_db, mock_sleep):
        req = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
        resp = json.dumps({
            "id": "abc-123",
            "model": "qwen",
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        })
        trace = _make_trace(req, resp)
        # First fetch is the startup bootstrap (nothing yet); second is the real poll.
        mock_fetch.side_effect = [[], [trace]]

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            localai_trace_polling_thread()

        mock_cursor.execute.assert_called_once()
        params = mock_cursor.execute.call_args[0][1]
        assert params[1] == "qwen"
        assert params[2] == 5
        assert params[3] == 10
        assert params[4] == 15
        assert params[6] == "hello"
        assert params[7] == "hi there"

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.get_db_connection")
    @patch("gpu_llm_monitor.fetch_traces")
    @patch.dict(os.environ, {"LOCALAI_TRACES_URL": "http://localai:4012/api/traces"})
    def test_bootstrap_skips_traces_already_in_buffer(self, mock_fetch, mock_db, mock_sleep):
        req = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
        resp = json.dumps({
            "id": "already-seen",
            "model": "qwen",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        trace = _make_trace(req, resp)
        # Same trace present at startup and on the first real poll -> must not be inserted.
        mock_fetch.side_effect = [[trace], [trace]]

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            localai_trace_polling_thread()

        mock_db.assert_not_called()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.fetch_traces")
    @patch.dict(os.environ, {"LOCALAI_TRACES_URL": "http://localai:4012/api/traces"})
    def test_fetch_failure_retries(self, mock_fetch, mock_sleep):
        mock_fetch.side_effect = [[], Exception("network down")]

        call_count = [0]

        def sleep_side_effect(s):
            call_count[0] += 1
            if call_count[0] >= 1:
                raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_side_effect

        with pytest.raises(KeyboardInterrupt):
            localai_trace_polling_thread()

        assert mock_fetch.call_count == 2
