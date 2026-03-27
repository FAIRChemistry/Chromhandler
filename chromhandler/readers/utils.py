from __future__ import annotations

import re

_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)[\s_]*(min|sec|h)\b", re.IGNORECASE)


def parse_reaction_time(stem: str) -> float:
    """Extract a reaction time (in minutes) from a filename stem.

    Supported units: ``min`` → as-is, ``sec`` → /60, ``h`` → x60.
    The *last* match in the stem is used so that prefixes like ``CV10_`` are
    ignored when the time is at the end.

    Args:
        stem: Filename without extension, e.g. ``"CV10_120min"``.

    Returns:
        Reaction time in minutes.

    Raises:
        ValueError: If no time pattern is found in *stem*.
    """
    matches = _TIME_RE.findall(stem)
    if not matches:
        raise ValueError(
            f"Cannot extract reaction time from filename stem '{stem}'. "
            "Expected a pattern like '30min', '120sec', or '2h'."
        )
    value_str, unit = matches[-1]
    value = float(value_str)
    match unit.lower():
        case "min":
            return value
        case "sec":
            return value / 60.0
        case "h":
            return value * 60.0
        case _:
            raise ValueError(f"Unrecognised time unit '{unit}' in stem '{stem}'.")
