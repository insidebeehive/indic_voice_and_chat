"""alembic/versions/0025_deposit_verification_order_idx.py

Adds a composite index on deposit_verification_requests(tenant_id, order_id).

POST /deposit-verification/reply/{token} -- the json_ticket_relay vendor's
ticket-reply callback -- has no request_id to look a row up by, so it queries
on tenant_id + order_id instead (deposit_ticket_reply in
src/api/deposit_verification.py):

    select(DepositVerificationRequest).where(
        DepositVerificationRequest.tenant_id == tenant.id,
        DepositVerificationRequest.order_id == body.order_id,
    ).order_by(DepositVerificationRequest.created_at.desc(),
               DepositVerificationRequest.id.desc()).limit(1)

The table's only index until now, idx_deposit_verification_requests_tenant_status_timeout
on (tenant_id, status, timeout_at), serves the timeout sweep and doesn't cover
this lookup -- order_id isn't in it, and status is deliberately not filtered
by this query -- so every vendor reply did a scan. This vendor sends multiple
messages per order (an agent_reply trail plus holding messages), so the query
runs repeatedly per dispute, not once.

Column order (tenant_id, order_id): both are equality predicates, and
tenant_id first matches the existing index's leading column and the table's
tenant-scoped access pattern. Deliberately NOT including created_at: the
(tenant_id, order_id) pair already narrows to ~1 row in practice, so the
ORDER BY created_at DESC, id DESC ... LIMIT 1 tiebreak (which exists only for
same-second created_at on second-granularity backends -- see the comment on
that query in src/api/deposit_verification.py) has nothing meaningful left to
sort.
A third column would add write cost for no read benefit.

Revision: 0025_deposit_verification_order_idx
Down: 0024_chat_msg_source_media_url
"""

from __future__ import annotations

from alembic import op

revision = "0025_deposit_verification_order_idx"
down_revision = "0024_chat_msg_source_media_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_deposit_verification_requests_tenant_order",
        "deposit_verification_requests",
        ["tenant_id", "order_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_deposit_verification_requests_tenant_order",
        table_name="deposit_verification_requests",
    )
