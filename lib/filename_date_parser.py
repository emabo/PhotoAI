#!/usr/bin/env python3
import re
from datetime import date, datetime, timezone
from typing import Optional, Tuple


FILENAME_DATE_PATTERNS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y.%m.%d%*",
    "%Y%m%d%H%M%S",
    "IMG-%Y%m%d-WA%f",
    "IMG-%Y%m%d-WA%f_01",
    "IMG-%Y%m%d-WA%f_1",
    "IMG-%Y%m%d-WA%f_01_01",
    "PANO_%Y%m%d_%H%M%S",
    "IMG_%Y%m%d_%H%M%S",
    "IMG_%Y-%m-%d-%f",
    "%Y%m%d_%H%M%S",
    "VID-%Y%m%d-WA%f",
    "VID_%Y%m%d_%H%M%S",
    "%Y%m%d_%H%M%S_%f",
    "%Y%m%d-WA%f",
    "%Y-%m-%d %H.%M.%S",
]


_TOKEN_REGEX = {
    "%Y": r"(?P<Y>\d{4})",
    "%m": r"(?P<m>\d{2})",
    "%d": r"(?P<d>\d{2})",
    "%H": r"(?P<H>\d{2})",
    "%M": r"(?P<M>\d{2})",
    "%S": r"(?P<S>\d{2})",
    "%f": r"(?P<f>\d{1,9})",
    "%*": r"(?:[._\s-].*)?",
}


def _compile_parser(pattern: str) -> re.Pattern:
    parts = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "%" and i + 1 < len(pattern):
            token = pattern[i : i + 2]
            if token in _TOKEN_REGEX:
                parts.append(_TOKEN_REGEX[token])
                i += 2
                continue
        parts.append(re.escape(pattern[i]))
        i += 1
    return re.compile("^" + "".join(parts) + "$")


_PARSER_REGEX = tuple((pattern, _compile_parser(pattern)) for pattern in FILENAME_DATE_PATTERNS)

# Detect embedded Unix timestamps: 13-digit (ms) tried first, then 10-digit (seconds)
_UNIX_MS_RE = re.compile(r"(?<!\d)(\d{13})(?!\d)")
_UNIX_S_RE  = re.compile(r"(?<!\d)(\d{10})(?!\d)")
_UNIX_YEAR_MIN = 946684800      # 2000-01-01 in seconds
_UNIX_YEAR_MAX = 4102444800     # 2100-01-01 in seconds


def _parse_unix_ts_from_stem(stem: str) -> Tuple[Optional[date], Optional[str]]:
    """Look for an embedded Unix timestamp (ms or s) anywhere in the filename stem."""
    for regex, divisor, label in (
        (_UNIX_MS_RE, 1000, "unix_ts_ms"),
        (_UNIX_S_RE,  1,    "unix_ts_s"),
    ):
        for m in regex.finditer(stem):
            ts_s = int(m.group(1)) / divisor
            if _UNIX_YEAR_MIN <= ts_s <= _UNIX_YEAR_MAX:
                try:
                    d = datetime.fromtimestamp(ts_s, tz=timezone.utc).date()
                    return d, label
                except (OSError, OverflowError, ValueError):
                    continue
    return None, None


def parse_date_from_stem(stem: str) -> Tuple[Optional[date], Optional[str]]:
    for pattern, regex in _PARSER_REGEX:
        match = regex.match(stem)
        if not match:
            continue
        try:
            year = int(match.group("Y"))
            month = int(match.group("m"))
            day = int(match.group("d"))
            return date(year, month, day), pattern
        except ValueError:
            continue
    # Fallback: embedded Unix timestamp (GoPro, etc.)
    return _parse_unix_ts_from_stem(stem)


def parse_datetime_from_stem(stem: str) -> Tuple[Optional[datetime], Optional[str]]:
    for pattern, regex in _PARSER_REGEX:
        match = regex.match(stem)
        if not match:
            continue
        try:
            year = int(match.group("Y"))
            month = int(match.group("m"))
            day = int(match.group("d"))

            h_raw = match.groupdict().get("H")
            m_raw = match.groupdict().get("M")
            s_raw = match.groupdict().get("S")

            hour = int(h_raw) if h_raw is not None else 0
            minute = int(m_raw) if m_raw is not None else 0
            second = int(s_raw) if s_raw is not None else 0

            return datetime(year, month, day, hour, minute, second), pattern
        except ValueError:
            continue

    parsed_date, pattern = _parse_unix_ts_from_stem(stem)
    if parsed_date is not None:
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0), pattern

    return None, None
