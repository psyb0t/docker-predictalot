from __future__ import annotations

import pytest

from predictalot.size import SizeParseError, parse_size


class TestParseSize:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("0", 0),
            ("1024", 1024),
            ("32mb", 32 * 1024 * 1024),
            ("32MB", 32 * 1024 * 1024),
            ("32m", 32 * 1024 * 1024),
            ("512k", 512 * 1024),
            ("512KB", 512 * 1024),
            ("1g", 1024 * 1024 * 1024),
            ("1gb", 1024 * 1024 * 1024),
            ("1GB", 1024 * 1024 * 1024),
            ("1b", 1),
            ("1B", 1),
        ],
    )
    def test_valid(self, s: str, expected: int) -> None:
        assert parse_size(s) == expected

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "1.5mb",
            "abc",
            "-1mb",
            "1tb",  # no T unit
            "1 mb",  # whitespace inside
            "mb",  # no value
        ],
    )
    def test_invalid(self, s: str) -> None:
        with pytest.raises(SizeParseError):
            parse_size(s)

    def test_strip_whitespace(self) -> None:
        assert parse_size("  32mb  ") == 32 * 1024 * 1024
