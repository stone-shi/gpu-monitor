import os
import sys
import pytest


def format_human(n):
    if n is None:
        return "0"
    n = float(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return f"{int(n)}"


class TestFormatHuman:
    def test_none(self):
        assert format_human(None) == "0"

    def test_zero(self):
        assert format_human(0) == "0"

    def test_small_number(self):
        assert format_human(42) == "42"

    def test_thousands(self):
        assert format_human(1500) == "1.5K"

    def test_millions(self):
        assert format_human(2500000) == "2.50M"

    def test_billions(self):
        assert format_human(3000000000) == "3.00B"

    def test_exact_thousand(self):
        assert format_human(1000) == "1.0K"

    def test_exact_million(self):
        assert format_human(1000000) == "1.00M"

    def test_exact_billion(self):
        assert format_human(1000000000) == "1.00B"

    def test_negative(self):
        assert format_human(-5000) == "-5.0K"

    def test_float_input(self):
        assert format_human(999.9) == "999"

    def test_large_thousands(self):
        assert format_human(999999) == "1000.0K"
