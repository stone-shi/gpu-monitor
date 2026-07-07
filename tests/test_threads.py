import os
import sys
import time
import json
import collections
import pytest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_llm_monitor import hardware_polling_thread, lms_log_processing_thread


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


class TestLmsLogProcessingThread:
    def _make_mock_proc(self, lines):
        proc = MagicMock()
        proc.stdout.readline.side_effect = lines + [""]
        proc.terminate.return_value = None
        proc.wait.return_value = None
        proc.kill.return_value = None
        return proc

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.subprocess.Popen")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_output_event_with_stats(self, mock_db, mock_popen, mock_sleep):
        log_line = json.dumps({
            "data": {
                "type": "llm.prediction.output",
                "modelIdentifier": "test-model",
                "stats": {
                    "promptTokensCount": 10,
                    "predictedTokensCount": 20,
                    "totalTokensCount": 30,
                    "tokensPerSecond": 15.5,
                },
                "output": "response text",
            }
        })

        proc = self._make_mock_proc([log_line])
        mock_popen.return_value = proc

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def sleep_side_effect(s):
            call_count[0] += 1
            if call_count[0] >= 1:
                raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_side_effect

        with pytest.raises(KeyboardInterrupt):
            lms_log_processing_thread()

        mock_cursor.execute.assert_called()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.subprocess.Popen")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_input_then_output_pairing(self, mock_db, mock_popen, mock_sleep):
        input_line = json.dumps({
            "data": {
                "type": "llm.prediction.input",
                "input": "hello world",
                "modelIdentifier": "my-model",
            }
        })
        output_line = json.dumps({
            "data": {
                "type": "llm.prediction.output",
                "modelIdentifier": "my-model",
                "stats": {
                    "promptTokensCount": 5,
                    "predictedTokensCount": 10,
                    "totalTokensCount": 15,
                    "tokensPerSecond": 25.0,
                },
                "output": "hi there",
            }
        })

        proc = self._make_mock_proc([input_line, output_line])
        mock_popen.return_value = proc

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            lms_log_processing_thread()

        insert_call = mock_cursor.execute.call_args
        params = insert_call[0][1]
        assert params[0] == "my-model"
        assert params[1] == 5
        assert params[2] == 10
        assert params[3] == 15
        assert params[4] == 25.0
        assert params[5] == "hello world"
        assert params[6] == "hi there"

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.subprocess.Popen")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_truncated_output_line(self, mock_db, mock_popen, mock_sleep):
        truncated = '{"data":{"type":"llm.prediction.output","output":"some truncated text'

        proc = self._make_mock_proc([truncated])
        mock_popen.return_value = proc

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_db.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            lms_log_processing_thread()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.subprocess.Popen")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_empty_and_non_json_lines_skipped(self, mock_db, mock_popen, mock_sleep):
        proc = self._make_mock_proc(["", "not json at all", "   "])
        mock_popen.return_value = proc

        mock_sleep.side_effect = lambda s: (_ for _ in ()).throw(KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            lms_log_processing_thread()

    @patch("gpu_llm_monitor.time.sleep")
    @patch("gpu_llm_monitor.subprocess.Popen")
    @patch("gpu_llm_monitor.get_db_connection")
    def test_subprocess_failure_restarts(self, mock_db, mock_popen, mock_sleep):
        mock_popen.side_effect = [Exception("spawn failed"), MagicMock()]

        call_count = [0]

        def sleep_side_effect(s):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise KeyboardInterrupt("stop")

        mock_sleep.side_effect = sleep_side_effect

        with pytest.raises(KeyboardInterrupt):
            lms_log_processing_thread()

        assert mock_popen.call_count >= 2
