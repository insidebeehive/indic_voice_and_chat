"""alembic/versions/0024_chat_msg_source_media_url.py

Adds ``source_media_url`` to ``chat_messages``: the ORIGINAL client-supplied
URL for an inbound image/video/audio message, when the client sent one
(e.g. the CRM's ``media_url``), kept alongside the ``media_url`` column we
already write with OUR OWN media-store object key for that same row.

Why keep both: today the CRM hands us a public https URL, we fetch it and
re-upload the bytes under our own key, and only that key survives — the
CRM's original URL is discarded once the re-upload succeeds. The
``json_ticket_relay`` deposit-verification vendor contract
(``src/chatbot/deposit_verification.py``) needs a publicly-fetchable https
URL to hand to the vendor, and today mints one from our own media store via
``IMediaStorage.signed_url()``. Persisting the CRM's original URL lets that
contract forward it directly instead, so a verification request no longer
depends on our media store being reachable.

NULLABLE, no server default: NULL means either the client uploaded raw
bytes (base64 ``data``, no URL to keep) or the row predates this column —
both are the status quo, nothing to backfill.

``Text``, not ``String(500)`` like ``media_url``: this holds a CRM/vendor
URL we don't control the length of (presigned query strings in particular
can run long), and inheriting ``media_url``'s 500-char cap here would risk
silently truncating or erroring on exactly the URLs this column exists to
preserve.

Revision: 0024_chat_msg_source_media_url
Down: 0023_chat_tool_result_chars

Note: chained after 0023_chat_tool_result_chars (not 0022 as originally
scoped) — 0023 landed as the actual current head by the time this migration
was written, and forking a second "0023" off 0022 would have left two heads
in the alembic history.

The id is abbreviated ("chat_msg", not "chat_message") to stay inside
``alembic_version.version_num``, which is ``varchar(32)`` — the unabridged
name was 34 characters and failed the version stamp at the end of an
otherwise successful upgrade.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_chat_msg_source_media_url"
down_revision = "0023_chat_tool_result_chars"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("source_media_url", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "source_media_url")
