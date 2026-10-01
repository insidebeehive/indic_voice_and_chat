"""Operator-posted hot-issue notices — live incident/outage/delay notices
injected directly into the chat/voice system prompt while active, instead of
living in the RAG KB. See ``src/chatbot/hot_issues.py``'s module docstring
for the full rationale (chat only reaches the KB through a tool call so a
notice isn't guaranteed to be seen; voice truncates its KB block to 15k
characters so a new notice can be cut; chunking/embedding a short,
hourly-changing notice is wasted cost).

One row is scoped to EXACTLY one of a tenant or a CRM, never both and never
neither (``ck_hot_issues_one_scope`` below) — a tenant's own notice
(``tenant_id`` set, ``crm_id`` NULL) or a CRM-wide notice shared by every
tenant linked to that CRM (``crm_id`` set, ``tenant_id`` NULL). A chat/call
sees the union of its tenant's rows and its linked CRM's rows
(``src/chatbot/hot_issues.py``'s ``get_active_hot_issues``), tenant issues
rendered first.

``key`` is the caller's stable identifier for one notice (e.g. "pg-delay"),
unique within its scope (``uq_hot_issues_tenant_key`` /
``uq_hot_issues_crm_key``) — stable across a PUT replace, so re-posting the
same incident under the same key updates it in place instead of duplicating
it. Must match ``HOT_ISSUE_KEY_RE`` below.

Caps (count per scope, title/body length, total chars per scope, the key
regex, duplicate keys, the expiry window) are enforced by the API
(``src/api/hot_issues.py``) with a 422, not by a DB constraint — the named
constants below exist so the model and the API can't silently drift on the
numbers.

Datetimes are naive UTC, matching ``src/models/chat.py``. ``expires_at``
defaults to ``HOT_ISSUE_DEFAULT_TTL_HOURS`` from now at write time (applied
by the API, not here), so re-PUTting an issue without one resets its expiry.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base

# Stable, log/URL-safe identifier for one notice: lowercase letters/digits,
# then up to 63 more of letters/digits/underscore/dot/hyphen (max 64 chars
# total). Must start with a letter or digit so it can never collide with a
# scope-qualifier prefix such as "t:"/"c:" (src/chatbot/hot_issues.py).
HOT_ISSUE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

# Caps enforced by the API (src/api/hot_issues.py) — see that module's own
# validation helper. Restated here, not re-derived, so the model and the API
# can't drift on the numbers.
HOT_ISSUE_MAX_PER_SCOPE = 5
HOT_ISSUE_TITLE_MAX_LEN = 80
HOT_ISSUE_BODY_MAX_LEN = 400
# Keeps the union of tenant + CRM issues at or under 1,500 chars by
# construction (5 issues/scope x 2 scopes x up to 750/scope) — no trim step
# needed downstream.
HOT_ISSUE_TOTAL_CHARS_MAX_PER_SCOPE = 750
HOT_ISSUE_DEFAULT_TTL_HOURS = 24
HOT_ISSUE_MAX_TTL_DAYS = 7


class HotIssue(Base):
    __tablename__ = "hot_issues"
    __table_args__ = (
        # The first CHECK constraint in this repo — valid on both SQLite and
        # Postgres. Enforces the one-scope rule at the DB level as a backstop
        # to the API's own tenant-XOR-crm construction; SQLite doesn't
        # enforce foreign keys by default, so this CHECK is the thing that
        # actually guards the invariant in the sqlite-backed unit-test suite.
        CheckConstraint(
            "(tenant_id IS NULL) <> (crm_id IS NULL)", name="ck_hot_issues_one_scope",
        ),
        # NULL is never considered equal to NULL by a unique index (both
        # SQLite and Postgres), so a crm-scoped row (tenant_id NULL) never
        # collides with another crm-scoped row under this constraint, and
        # vice versa for the crm_id constraint below — each constraint only
        # ever actually bites within its own scope.
        UniqueConstraint("tenant_id", "key", name="uq_hot_issues_tenant_key"),
        UniqueConstraint("crm_id", "key", name="uq_hot_issues_crm_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # No index=True here: each UNIQUE constraint below already leads with this
    # same column (uq_hot_issues_tenant_key / uq_hot_issues_crm_key), so a
    # separate single-column index would be redundant -- and the migration
    # (alembic/versions/0030_hot_issues.py) deliberately doesn't create one
    # either, so autogenerate sees no drift between the two.
    tenant_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
    )
    crm_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"),
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now())
