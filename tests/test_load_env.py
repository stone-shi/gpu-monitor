import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_llm_monitor import load_env


class TestLoadEnv:
    def test_missing_file(self):
        load_env("/nonexistent/path/.env")

    def test_basic_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")
        load_env(str(env_file))
        assert os.environ["FOO"] == "bar"
        assert os.environ["BAZ"] == "qux"
        del os.environ["FOO"]
        del os.environ["BAZ"]

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        load_env(str(env_file))
        assert os.environ["KEY"] == "val"
        del os.environ["KEY"]

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("A='single'\nB=\"double\"\n")
        load_env(str(env_file))
        assert os.environ["A"] == "single"
        assert os.environ["B"] == "double"
        del os.environ["A"]
        del os.environ["B"]

    def test_value_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("CONN=host=localhost port=5432\n")
        load_env(str(env_file))
        assert os.environ["CONN"] == "host=localhost port=5432"
        del os.environ["CONN"]

    def test_whitespace_trimming(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("  KEY  =  value  \n")
        load_env(str(env_file))
        assert os.environ["KEY"] == "value"
        del os.environ["KEY"]
