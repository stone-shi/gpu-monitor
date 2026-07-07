import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_llm_monitor import get_db_connection


class TestGetDbConnection:
    @patch("gpu_llm_monitor.psycopg2.connect")
    def test_default_values(self, mock_connect):
        mock_connect.return_value = MagicMock()
        env = {
            "POSTGRES_USER": "",
            "POSTGRES_PASSWORD": "",
            "DB_HOST": "",
            "DB_PORT": "",
            "DB_NAME": "",
        }
        with patch.dict(os.environ, env, clear=False):
            for k in env:
                os.environ.pop(k, None)
            get_db_connection()
            mock_connect.assert_called_once_with(
                host="localhost",
                port="5432",
                user="postgres",
                password="",
                database="gpu-monitor",
            )

    @patch("gpu_llm_monitor.psycopg2.connect")
    def test_custom_values(self, mock_connect):
        mock_connect.return_value = MagicMock()
        env = {
            "POSTGRES_USER": "myuser",
            "POSTGRES_PASSWORD": "mypass",
            "DB_HOST": "dbhost",
            "DB_PORT": "5433",
            "DB_NAME": "mydb",
        }
        with patch.dict(os.environ, env):
            get_db_connection()
            mock_connect.assert_called_once_with(
                host="dbhost",
                port="5433",
                user="myuser",
                password="mypass",
                database="mydb",
            )

    @patch("gpu_llm_monitor.psycopg2.connect")
    def test_connection_returned(self, mock_connect):
        sentinel = MagicMock()
        mock_connect.return_value = sentinel
        with patch.dict(os.environ, {}, clear=False):
            result = get_db_connection()
            assert result is sentinel
