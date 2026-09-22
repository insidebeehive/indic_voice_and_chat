"""Unit tests for the BO support-hours schedule parser.

Fix 2: ``_parse_schedule`` silently drops an entry on two typo shapes -- an
unrecognized day-range key (``mon-fry`` instead of ``mon-fri``) and an
unparseable "HH:MM-HH:MM" value (``9am-6pm``) -- with only a DEBUG event (off
in normal running). One mistyped character can delete an entire working
week's worth of availability with no signal at any level: backoffice then
reads unavailable Monday-Friday and human handover is never offered.

These tests pin two things at once, deliberately: (1) the drop now WARNS, and
(2) the parsing behaviour itself is completely unchanged -- this is a
visibility fix only, so the "returned schedule is identical to before" half
of each test is the regression guard against ever tightening the parsing as
a side effect.
"""

from __future__ import annotations

import logging

from src.chatbot.support_hours import _parse_schedule


def test_all_valid_keys_parse_with_no_warning(caplog):
    """Baseline: a fully valid config parses all 6 days and warns nothing --
    proves the two typo tests below are actually exercising the drop path,
    not just an unrelated difference in fixture shape."""
    with caplog.at_level(logging.WARNING, logger="src.chatbot.support_hours"):
        schedule = _parse_schedule({"mon-fri": "09:00-18:00", "sat": "10:00-14:00"})
    assert len(schedule) == 6  # mon,tue,wed,thu,fri,sat
    assert schedule[0] == (9 * 60, 18 * 60)
    assert schedule[5] == (10 * 60, 14 * 60)
    assert caplog.records == []


def test_unrecognized_day_key_warns_and_drops_only_that_entry(caplog):
    """'mon-fry' (typo for 'mon-fri') matches no entry in _DAY_RANGES, so the
    whole Mon-Fri working week is dropped -- 'sat' is unaffected and still
    parses. The warning must name the offending key and value so an operator
    can find the typo without reading source."""
    with caplog.at_level(logging.WARNING, logger="src.chatbot.support_hours"):
        schedule = _parse_schedule({"mon-fry": "09:00-18:00", "sat": "10:00-14:00"})

    # Behaviour unchanged from before this fix: Saturday only, Mon-Fri gone.
    assert schedule == {5: (10 * 60, 14 * 60)}

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "mon-fry" in msg
    assert "09:00-18:00" in msg


def test_unparseable_timerange_warns_and_drops_only_that_entry(caplog):
    """'9am-6pm' isn't "HH:MM-HH:MM", so int() parsing raises and the entry is
    dropped -- with a valid day key this time, so the failure is purely in the
    time-range parsing branch, not the day-key branch above."""
    with caplog.at_level(logging.WARNING, logger="src.chatbot.support_hours"):
        schedule = _parse_schedule({"mon-fri": "9am-6pm"})

    # Behaviour unchanged from before this fix: nothing configured at all.
    assert schedule == {}

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "mon-fri" in msg
    assert "9am-6pm" in msg


def test_typo_that_deletes_the_working_week_is_now_visible(caplog):
    """The exact demonstrated regression from the report: a single mistyped
    character in the day-range key silently removed 5 of 6 configured days.
    Behaviour is unchanged (still only Saturday) but it is no longer silent."""
    with caplog.at_level(logging.WARNING, logger="src.chatbot.support_hours"):
        schedule = _parse_schedule({"mon-fry": "09:00-18:00", "sat": "10:00-14:00"})
    assert len(schedule) == 1
    assert len(caplog.records) == 1
