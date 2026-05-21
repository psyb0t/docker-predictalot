from __future__ import annotations

import pytest

from predictalot.duration import DurationParseError, parse_duration


class TestParseDuration:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("0", 0.0),
            ("30s", 30.0),
            ("5m", 300.0),
            ("1h", 3600.0),
            ("2h30m", 2 * 3600 + 30 * 60),
            ("1d", 86400.0),
            ("1d2h3m4s", 86400 + 2 * 3600 + 3 * 60 + 4),
            ("90m", 5400.0),
        ],
    )
    def test_valid(self, s: str, expected: float) -> None:
        assert parse_duration(s) == expected

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "5",  # no unit
            "5min",  # unsupported unit
            "1y",  # no year unit
            "1.5h",  # no fractional
            "abc",
            "1h2x",
            "h",  # no value
            "-5m",  # negative
            "1d 2h",  # whitespace inside
        ],
    )
    def test_invalid(self, s: str) -> None:
        with pytest.raises(DurationParseError):
            parse_duration(s)

    def test_strip_whitespace(self) -> None:
        assert parse_duration("  1h  ") == 3600.0
