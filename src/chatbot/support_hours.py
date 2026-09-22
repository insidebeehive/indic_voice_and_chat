"""BO support-hours availability checker."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.utils.logging import debug_event

log = logging.getLogger(__name__)

_DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

_DAY_RANGES: dict = {
    "mon-fri": [0, 1, 2, 3, 4],
    "weekdays": [0, 1, 2, 3, 4],
    "weekend": [5, 6],
    **{d: [i] for i, d in enumerate(_DAY_NAMES)},
}


def _parse_schedule(support_hours: dict) -> dict:
    """Expand support_hours dict to {weekday_int: (start_min, end_min)}."""
    schedule: dict = {}
    for key, timerange in support_hours.items():
        if not timerange:
            continue
        days = _DAY_RANGES.get(key.lower(), [])
        if not days:
            # A key that doesn't match any known day range (typo, e.g.
            # "mon-fry") silently drops this whole entry -- the `for d in
            # days` loop below would just no-op on an empty list with no
            # trace that the tenant's configured hours for that key were
            # ever read. This is the entry that made a tenant's support
            # hours look "always closed" for one day with no error anywhere.
            debug_event(
                log, "support_hours schedule_entry_dropped", key=key,
                timerange=timerange, reason="unrecognized_day_key",
            )
            continue
        try:
            start_s, end_s = timerange.split("-", 1)
            sh, sm = int(start_s.split(":")[0]), int(start_s.split(":")[1])
            eh, em = int(end_s.split(":")[0]), int(end_s.split(":")[1])
        except Exception:  # noqa: BLE001
            # A malformed "HH:MM-HH:MM" value (missing dash, non-numeric,
            # etc) is dropped the same silent way -- same motivation as above.
            debug_event(
                log, "support_hours schedule_entry_dropped", key=key,
                timerange=timerange, reason="unparseable_timerange",
            )
            continue
        for d in days:
            schedule[d] = (sh * 60 + sm, eh * 60 + em)
    return schedule


def is_bo_available(chat_support) -> tuple:
    """Return (available: bool, next_slot_description: str).

    If support_hours is empty, availability check is disabled → always available.
    next_slot_description is only meaningful when available=False.
    """
    support_hours = getattr(chat_support, "support_hours", {}) or {}
    if not support_hours:
        debug_event(
            log, "support_hours availability decision", available=True,
            reason="support_hours_check_disabled",
        )
        return True, ""

    tz_name = getattr(chat_support, "support_timezone", "Asia/Kolkata") or "Asia/Kolkata"
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        # Silent fallback: a mistyped/unknown IANA zone name in tenant config
        # would otherwise make every availability decision below silently
        # evaluate against the wrong timezone, with nothing to show it wasn't
        # the tenant's own configured one.
        debug_event(
            log, "support_hours timezone_resolve fallback",
            configured_timezone=tz_name, fallback_timezone="Asia/Kolkata",
        )
        tz = ZoneInfo("Asia/Kolkata")

    now = datetime.now(tz)
    schedule = _parse_schedule(support_hours)
    if not schedule:
        debug_event(
            log, "support_hours availability decision", available=True,
            reason="schedule_parse_empty", support_hours=support_hours,
        )
        return True, ""

    cur_day = now.weekday()  # 0=Mon
    cur_min = now.hour * 60 + now.minute

    # Check today
    if cur_day in schedule:
        start, end = schedule[cur_day]
        if start <= cur_min < end:
            debug_event(
                log, "support_hours availability decision", available=True,
                reason="within_today_window", now=now.isoformat(), timezone=tz_name,
                cur_day=cur_day, cur_min=cur_min, window=[start, end],
            )
            return True, ""

    # Find next available slot (search up to 7 days ahead)
    for delta in range(1, 8):
        next_day = (cur_day + delta) % 7
        if next_day in schedule:
            start, _ = schedule[next_day]
            next_dt = now + timedelta(days=delta)
            next_dt = next_dt.replace(
                hour=start // 60, minute=start % 60, second=0, microsecond=0)
            label = next_dt.strftime("%A, %d %b at %I:%M %p %Z")
            debug_event(
                log, "support_hours availability decision", available=False,
                reason="outside_hours", now=now.isoformat(), timezone=tz_name,
                cur_day=cur_day, cur_min=cur_min, next_slot=label,
            )
            return False, label

    debug_event(
        log, "support_hours availability decision", available=False,
        reason="no_slot_found_in_7_days", now=now.isoformat(), timezone=tz_name,
        schedule=schedule,
    )
    return False, "soon"
