"""Campaign domain models (pydantic).

Distinct from the SQLAlchemy ORM in ``src/models/campaign.py``: those are
the persisted shape, these are the API + business-logic shape. The two
share the same field semantics but live in different namespaces so we can
evolve the API surface without rippling through migrations.

Lead status state machine (PRD §6.1):
    pending  -> in_flight -> {completed, retry, dnd}
    retry    -> in_flight (after retry_interval_hours)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --- Enums ---------------------------------------------------------------


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class LeadStatus(str, Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    RETRY = "retry"
    DND = "dnd"
    FAILED = "failed"


class CallDisposition(str, Enum):
    INTERESTED_CALLBACK = "interested_callback"
    INTERESTED_TRANSFER = "interested_transfer"
    NOT_INTERESTED = "not_interested"
    BUSY_RETRY = "busy_retry"
    DND_REQUESTED = "dnd_requested"
    WRONG_NUMBER = "wrong_number"
    VOICEMAIL = "voicemail"


class LeadCallOutcome(str, Enum):
    INTERESTED = "interested"
    CALLBACK_REQUESTED = "callback_requested"
    NOT_INTERESTED = "not_interested"
    REFUSED = "refused"
    ESCALATED = "escalated"
    ANGRY_HOSTILE = "angry_hostile"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"
    BUSY = "busy"
    CALL_FAILED = "call_failed"
    RECORDING_UNAVAILABLE = "recording-unavailable"


# Conversational outcomes the LLM may return (the rest come from telephony).
CONVERSATIONAL_OUTCOMES = frozenset({
    LeadCallOutcome.INTERESTED,
    LeadCallOutcome.CALLBACK_REQUESTED,
    LeadCallOutcome.NOT_INTERESTED,
    LeadCallOutcome.REFUSED,
    LeadCallOutcome.ESCALATED,
    LeadCallOutcome.ANGRY_HOSTILE,
})

_OUTCOME_TO_DISPOSITION = {
    LeadCallOutcome.INTERESTED: CallDisposition.INTERESTED_TRANSFER,
    LeadCallOutcome.CALLBACK_REQUESTED: CallDisposition.INTERESTED_CALLBACK,
    LeadCallOutcome.NOT_INTERESTED: CallDisposition.NOT_INTERESTED,
    LeadCallOutcome.REFUSED: CallDisposition.DND_REQUESTED,
    LeadCallOutcome.ESCALATED: CallDisposition.INTERESTED_TRANSFER,
    LeadCallOutcome.ANGRY_HOSTILE: CallDisposition.DND_REQUESTED,
    LeadCallOutcome.NO_ANSWER: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.BUSY: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.CALL_FAILED: CallDisposition.BUSY_RETRY,
    LeadCallOutcome.VOICEMAIL: CallDisposition.VOICEMAIL,
    LeadCallOutcome.RECORDING_UNAVAILABLE: CallDisposition.BUSY_RETRY,
}

_TELEPHONY_TO_OUTCOME = {
    "no_answer": LeadCallOutcome.NO_ANSWER,
    "busy": LeadCallOutcome.BUSY,
    "failed": LeadCallOutcome.CALL_FAILED,
    "voicemail": LeadCallOutcome.VOICEMAIL,
}


def disposition_from_outcome(outcome: LeadCallOutcome) -> CallDisposition:
    """Map a canonical outcome to the legacy disposition consumed by the
    orchestrator/CRM/benchmarks. Total over LeadCallOutcome."""
    return _OUTCOME_TO_DISPOSITION[outcome]


def outcome_from_telephony(status: Optional[str]) -> Optional[LeadCallOutcome]:
    """Map a normalized telephony status to an unreachable outcome, or None
    when the call connected (the conversational path then applies)."""
    return _TELEPHONY_TO_OUTCOME.get(status or "")


# --- Models --------------------------------------------------------------


class CallAnalysis(BaseModel):
    """Result of analyzing one finished call. Produced by analyze_call()."""

    outcome: LeadCallOutcome
    summary: str = ""           # English, 2-3 sentences
    notes: str = ""             # English; objections, preferences, next steps
    callback_datetime: Optional[datetime] = None  # tz-aware when resolved
    callback_phrase: Optional[str] = None         # raw, e.g. "kal shaam 5 baje"
    analysis_source: Literal["llm", "telephony", "fallback"] = "llm"


class Lead(BaseModel):
    id: str
    tenant_id: str
    campaign_id: Optional[str] = None
    phone_number: str = Field(min_length=1)
    name: Optional[str] = None
    language_pref: Optional[str] = None
    crm_lead_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: LeadStatus = LeadStatus.PENDING
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("phone_number")
    @classmethod
    def _strip_phone(cls, v: str) -> str:
        return v.strip()


class Campaign(BaseModel):
    id: str
    tenant_id: str
    name: str = Field(min_length=1, max_length=255)
    status: CampaignStatus = CampaignStatus.DRAFT
    config_yaml: str = ""  # serialized campaign YAML (PRD §5.2)
    total_leads: int = 0
    calls_attempted: int = 0
    calls_answered: int = 0
    leads_qualified: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CallResult(BaseModel):
    """Outcome of one completed call. Fed back into CRM + event bus."""

    session_id: str
    tenant_id: str
    campaign_id: str
    lead_id: str
    disposition: CallDisposition
    interest_level: Optional[str] = None
    slots: dict[str, Any] = Field(default_factory=dict)
    outcome: Optional[LeadCallOutcome] = None
    summary: str = ""
    notes: str = ""
    callback_datetime: Optional[datetime] = None  # -> Conversation.callback_at (DB)
    duration_ms: int = 0
    total_turns: int = 0
    sentiment_history: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Lead import --------------------------------------------------------


class LeadImportError(ValueError):
    """Raised when a CSV row cannot be turned into a Lead."""


def parse_leads_csv(
    data: bytes,
    campaign_id: str,
    tenant_id: str,
    id_prefix: str = "lead",
) -> tuple[list[Lead], list[tuple[int, str]]]:
    """Parse a CSV blob into ``Lead`` objects.

    Required columns: ``phone_number``. Optional: ``name``, ``language_pref``,
    ``crm_lead_id``, ``id``. Any other columns land in ``metadata``.

    Returns ``(leads, errors)`` where ``errors`` is ``[(row_number, reason)]``.
    Row numbering is 1-based, matching what a spreadsheet user expects.
    """
    leads: list[Lead] = []
    errors: list[tuple[int, str]] = []
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "phone_number" not in (reader.fieldnames or []):
        raise LeadImportError("CSV must have a 'phone_number' column")

    seen_ids: set[str] = set()
    for row_num, row in enumerate(reader, start=2):  # row 1 is header
        try:
            phone = (row.get("phone_number") or "").strip()
            if not phone:
                errors.append((row_num, "missing phone_number"))
                continue
            lead_id = (row.get("id") or f"{id_prefix}_{campaign_id}_{row_num - 1:06d}").strip()
            if lead_id in seen_ids:
                errors.append((row_num, f"duplicate id '{lead_id}'"))
                continue
            seen_ids.add(lead_id)

            standard_fields = {"id", "phone_number", "name", "language_pref", "crm_lead_id"}
            metadata = {k: v for k, v in row.items() if k not in standard_fields and v}

            leads.append(Lead(
                id=lead_id,
                tenant_id=tenant_id,
                campaign_id=campaign_id,
                phone_number=phone,
                name=(row.get("name") or "").strip() or None,
                language_pref=(row.get("language_pref") or "").strip() or None,
                crm_lead_id=(row.get("crm_lead_id") or "").strip() or None,
                metadata=metadata,
            ))
        except Exception as e:  # noqa: BLE001
            errors.append((row_num, str(e)))
    return leads, errors


def leads_from_dicts(
    rows: list[dict[str, Any]],
    campaign_id: str,
    tenant_id: str,
    id_prefix: str = "lead",
) -> tuple[list[Lead], list[tuple[int, str]]]:
    """Build leads from a list of dicts (e.g. from a CRM API response)."""
    leads: list[Lead] = []
    errors: list[tuple[int, str]] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        try:
            phone = str(row.get("phone_number") or "").strip()
            if not phone:
                errors.append((i, "missing phone_number"))
                continue
            lead_id = str(row.get("id") or f"{id_prefix}_{campaign_id}_{i:06d}")
            if lead_id in seen_ids:
                errors.append((i, f"duplicate id '{lead_id}'"))
                continue
            seen_ids.add(lead_id)
            standard_fields = {"id", "phone_number", "name", "language_pref", "crm_lead_id"}
            metadata = {k: v for k, v in row.items() if k not in standard_fields and v}
            leads.append(Lead(
                id=lead_id,
                tenant_id=tenant_id,
                campaign_id=campaign_id,
                phone_number=phone,
                name=row.get("name"),
                language_pref=row.get("language_pref"),
                crm_lead_id=row.get("crm_lead_id"),
                metadata=metadata,
            ))
        except Exception as e:  # noqa: BLE001
            errors.append((i, str(e)))
    return leads, errors
