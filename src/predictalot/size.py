"""Human-friendly byte-size parser. Base-1024.

Supported units (case-insensitive): b, k/kb, m/mb, g/gb.
Plain integers without a unit are interpreted as raw bytes.
Examples: 32mb, 512k, 1g, 1024, 0.
"""

from __future__ import annotations

import re

_PATTERN = re.compile(r"^(\d+)(b|k|kb|m|mb|g|gb)?$", re.IGNORECASE)

_MULTIPLIER = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024 * 1024,
    "mb": 1024 * 1024,
    "g": 1024 * 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
}


class SizeParseError(ValueError):
    pass


def parse_size(s: str) -> int:
    """Parse a size string into bytes.

    Args:
        s: size string like "32mb", "512k", "1g", "1024", "0".

    Returns:
        Total bytes as an int.

    Raises:
        SizeParseError on bad input.
    """
    raw = s.strip()
    if raw == "":
        raise SizeParseError("size is empty")

    m = _PATTERN.match(raw)
    if not m:
        raise SizeParseError(
            f"invalid size {raw!r}; expected like 32mb, 512k, 1g, 1024, or 0"
        )

    value = int(m.group(1))
    unit = (m.group(2) or "").lower()
    return value * _MULTIPLIER[unit]
