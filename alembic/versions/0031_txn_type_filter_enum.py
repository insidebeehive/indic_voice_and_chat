"""alembic/versions/0031_txn_type_filter_enum.py

Data migration: corrects the ``type`` filter of the ``get_player_transactions``
tool in every ``chat_tools`` and ``crm_tools`` row. The CRM API's values are
deposit | withdraw | casino | sports (omit for all); the seeded rows said
"withdrawal" and "all" and made ``type`` required.

* ``parameters["type"]`` is overwritten with the new spec regardless of any
  custom text (its values are fixed by the CRM API), but only when the row
  already has a ``type`` parameter.
* ``description`` is replaced only when it equals one of three historical
  catalog texts exactly (commits 5ecdc36, 937c168, 495da20+); customised
  descriptions are left alone.

Downgrade restores the old ``type`` spec (only where a ``type`` key exists),
and, where the description equals the new one, the most recent old text
(the 495da20+ HEAD-era description); older variants are not reconstructed.

Revision: 0031_txn_type_filter_enum
Down: 0030_hot_issues
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0031_txn_type_filter_enum"
down_revision = "0030_hot_issues"
branch_labels = None
depends_on = None

TOOL_NAME = "get_player_transactions"
TABLES = ("chat_tools", "crm_tools")

_TAIL = (
    "NOTE: for a deposit DISPUTE, prefer get_player_latest_deposit_order "
    "instead: it targets the specific recent attempt in one call and "
    "exposes the pending/failed status detail a dispute needs."
)

_HEAD = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Supports filtering by type and "
    "date range via query params. Use to answer questions like 'did my "
    "deposit go through?' or 'show my recent withdrawals'."
)

# Description as of HEAD before this migration (commit 495da20 onward).
# Downgrade restores this one.
OLD_DESCRIPTION = _HEAD + " " + _TAIL

# Older variants that were seeded into DB rows historically.
OLD_DESCRIPTION_NO_NOTE = _HEAD  # commit 5ecdc36
OLD_DESCRIPTION_PGSORDER_NOTE = (  # commit 937c168
    _HEAD + " NOTE: this does NOT return the PgsOrderId needed to raise a "
    "deposit verification \u2014 call get_player_latest_deposit_order for that."
)
KNOWN_OLD_DESCRIPTIONS = (
    OLD_DESCRIPTION, OLD_DESCRIPTION_NO_NOTE, OLD_DESCRIPTION_PGSORDER_NOTE,
)

NEW_DESCRIPTION = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Most questions are about a "
    "deposit or a withdrawal: work out which, then filter with type "
    "(deposit or withdraw). If the message does not make clear whether it "
    "is a deposit or a withdrawal, ask the customer one short question "
    "before calling this. Use casino or sports only when the customer "
    "clearly asks about those; omit type only for a general request "
    "like 'show my recent transactions'. " + _TAIL
)

OLD_TYPE_SPEC = {
    "type": "string", "source": "llm",
    "description": "Filter: deposit | withdrawal | casino | sports | all (default: all)",
}

NEW_TYPE_SPEC = {
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdraw", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdraw for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}


def transform_row(
    parameters, description, *, from_desc, to_desc, to_type_spec,
) -> tuple[dict, str]:
    """Pure row transform: returns (new_parameters, new_description)."""
    params = dict(parameters or {})
    if "type" in params:
        params["type"] = dict(to_type_spec)
        if "enum" in params["type"]:
            params["type"]["enum"] = list(params["type"]["enum"])
    from_descs = (from_desc,) if isinstance(from_desc, str) else tuple(from_desc)
    new_desc = to_desc if description in from_descs else description
    return params, new_desc


def _apply(from_desc, to_desc, to_type_spec) -> None:
    bind = op.get_bind()
    for table_name in TABLES:
        if not sa.inspect(bind).has_table(table_name):
            continue
        t = sa.table(
            table_name,
            sa.column("id", sa.Integer()),
            sa.column("name", sa.String()),
            sa.column("description", sa.Text()),
            sa.column("parameters", sa.JSON()),
        )
        rows = bind.execute(
            sa.select(t.c.id, t.c.description, t.c.parameters)
            .where(t.c.name == TOOL_NAME)
        ).fetchall()
        for row_id, desc, params in rows:
            new_params, new_desc = transform_row(
                params, desc, from_desc=from_desc, to_desc=to_desc,
                to_type_spec=to_type_spec,
            )
            bind.execute(
                t.update().where(t.c.id == row_id)
                .values(parameters=new_params, description=new_desc)
            )


def upgrade() -> None:
    _apply(KNOWN_OLD_DESCRIPTIONS, NEW_DESCRIPTION, NEW_TYPE_SPEC)


def downgrade() -> None:
    _apply(NEW_DESCRIPTION, OLD_DESCRIPTION, OLD_TYPE_SPEC)
