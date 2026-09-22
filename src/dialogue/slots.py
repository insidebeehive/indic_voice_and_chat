"""Slot schema + filling logic.

A slot is a structured piece of information the agent wants to extract over
the course of a conversation (interest level, callback time, current
provider, etc). The schema is loaded from the campaign YAML; the LLM emits
``updated_slots`` per turn (PRD §12.2) and ``apply_updates`` validates +
merges those into the running slot store.

Validation is deliberately lenient — bad LLM output should be logged and
dropped, never raise.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.utils.logging import debug_event

log = logging.getLogger(__name__)


class SlotType(str, Enum):
    STRING = "string"
    PHONE = "phone"
    EMAIL = "email"
    DATETIME = "datetime"
    ENUM = "enum"
    NUMBER = "number"
    BOOLEAN = "boolean"


@dataclass
class SlotSpec:
    name: str
    type: SlotType
    required: bool = False
    values: Optional[list[str]] = None  # for ENUM
    source: Optional[str] = None  # "crm", "user", etc

    @classmethod
    def from_raw(cls, name: str, raw: dict[str, Any]) -> "SlotSpec":
        return cls(
            name=name,
            type=SlotType(raw.get("type", "string")),
            required=bool(raw.get("required", False)),
            values=list(raw["values"]) if raw.get("values") else None,
            source=raw.get("source"),
        )


@dataclass
class SlotSchema:
    specs: dict[str, SlotSpec] = field(default_factory=dict)

    @classmethod
    def from_campaign_yaml(cls, slots: dict[str, dict[str, Any]]) -> "SlotSchema":
        return cls(specs={name: SlotSpec.from_raw(name, raw) for name, raw in slots.items()})

    def required_names(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.required]


class SlotFiller:
    """Holds slot values for one conversation, validates updates from the LLM."""

    def __init__(self, schema: SlotSchema, initial: Optional[dict[str, Any]] = None) -> None:
        self.schema = schema
        self._values: dict[str, Any] = dict(initial or {})
        self._rejected: list[tuple[str, Any, str]] = []

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    @property
    def rejected(self) -> list[tuple[str, Any, str]]:
        """Reasons updates were rejected, for logging."""
        return list(self._rejected)

    def get(self, name: str) -> Any:
        return self._values.get(name)

    def apply_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Validate + merge ``updates``. Returns the slots actually applied."""
        applied: dict[str, Any] = {}
        # Runs on every turn that carries updated_slots, so anything built
        # purely for the debug line below is gated on the level check rather
        # than paid unconditionally (docs/debug-logging.md's cost rule) --
        # this module had zero logging before this pass, so a rejected slot
        # (bad LLM output, silently dropped per this file's own docstring)
        # left no trace at any level.
        debug_on = log.isEnabledFor(logging.DEBUG)
        previous: dict[str, Any] = {}
        rejected_before = len(self._rejected)
        for name, raw_value in (updates or {}).items():
            spec = self.schema.specs.get(name)
            if spec is None:
                self._rejected.append((name, raw_value, "unknown slot"))
                continue
            if raw_value is None or raw_value == "":
                continue
            ok, coerced, reason = _validate(spec, raw_value)
            if not ok:
                self._rejected.append((name, raw_value, reason))
                continue
            if debug_on:
                previous[name] = self._values.get(name)
            self._values[name] = coerced
            applied[name] = coerced
        if debug_on:
            debug_event(
                log, "slots apply_updates completed",
                updates=dict(updates or {}), applied=applied,
                previous_values=previous, rejected=self._rejected[rejected_before:],
            )
        return applied

    def missing_required(self) -> list[str]:
        return [n for n in self.schema.required_names() if n not in self._values]

    def is_complete(self) -> bool:
        return not self.missing_required()


# --- validation -----------------------------------------------------------


_PHONE_RE = re.compile(r"^[+]?[0-9 \-()]{6,20}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate(spec: SlotSpec, value: Any) -> tuple[bool, Any, str]:
    t = spec.type
    if t is SlotType.STRING:
        return True, str(value), ""
    if t is SlotType.NUMBER:
        try:
            return True, float(value), ""
        except (TypeError, ValueError):
            return False, value, "not a number"
    if t is SlotType.BOOLEAN:
        if isinstance(value, bool):
            return True, value, ""
        if isinstance(value, str) and value.lower() in {"true", "yes", "haan", "1"}:
            return True, True, ""
        if isinstance(value, str) and value.lower() in {"false", "no", "nahi", "0"}:
            return True, False, ""
        return False, value, "not a boolean"
    if t is SlotType.ENUM:
        if spec.values and str(value) in spec.values:
            return True, str(value), ""
        return False, value, f"not in allowed values {spec.values}"
    if t is SlotType.PHONE:
        if isinstance(value, str) and _PHONE_RE.match(value.strip()):
            return True, value.strip(), ""
        return False, value, "invalid phone format"
    if t is SlotType.EMAIL:
        if isinstance(value, str) and _EMAIL_RE.match(value.strip()):
            return True, value.strip(), ""
        return False, value, "invalid email format"
    if t is SlotType.DATETIME:
        if isinstance(value, datetime):
            return True, value.isoformat(), ""
        if isinstance(value, str):
            try:
                return True, datetime.fromisoformat(value).isoformat(), ""
            except ValueError:
                # Accept opaque strings — TTS / CRM may parse later. Best-effort.
                return True, value, ""
        return False, value, "invalid datetime"
    return False, value, f"unknown slot type {t}"
