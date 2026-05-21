"""Go-style human-friendly duration parser.

Supported units: s, m, h, d (d = 24h).
Valid examples: 30s, 5m, 1h, 2h30m, 1d, 1d2h3m4s, 90m, 0.
"""

from __future__ import annotations

import re

_PATTERN = re.compile(
    r"^(?:0|(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?)$"
)

_UNIT_SECONDS = {
    "d": 86400.0,
    "h": 3600.0,
    "m": 60.0,
    "s": 1.0,
}


class DurationParseError(ValueError):
    pass


def parse_duration(s: str) -> float:
    """Parse a duration string into seconds.

    Args:
        s: duration string like "30s", "2h30m", "1d2h3m4s", or "0".

    Returns:
        Total seconds as a float.

    Raises:
        DurationParseError if the string doesn't match the grammar or yields 0
        from a non-"0" input (e.g. empty units like "h" or "1xyz").
    """
    raw = s.strip()
    if raw == "":
        raise DurationParseError("duration is empty")
    if raw == "0":
        return 0.0

    m = _PATTERN.match(raw)
    if not m or not any(m.groups()):
        raise DurationParseError(
            f"invalid duration {raw!r}; expected like 30s, 5m, 1h, 2h30m, 1d2h3m4s, or 0"
        )

    days, hours, minutes, seconds = m.groups()
    total = 0.0
    for value, unit in (
        (days, "d"),
        (hours, "h"),
        (minutes, "m"),
        (seconds, "s"),
    ):
        if value is not None:
            total += int(value) * _UNIT_SECONDS[unit]
    return total
