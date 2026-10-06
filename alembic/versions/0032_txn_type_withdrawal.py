"""alembic/versions/0032_txn_type_withdrawal.py

Data migration: the CRM API's real ``type`` value for withdrawals is
``withdrawal`` (0031 wrote ``withdraw``). For every ``get_player_transactions``
row in ``chat_tools`` and ``crm_tools``:

* ``parameters["type"]`` is replaced only when it equals 0031's spec exactly;
  any other (custom) spec is left alone.
* ``description`` is replaced only when it equals 0031's text exactly.

Downgrade reverses both (withdrawal text/spec back to the 0031 text/spec).

Revision: 0032_txn_type_withdrawal
Down: 0031_txn_type_filter_enum
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0032_txn_type_withdrawal"
down_revision = "0031_txn_type_filter_enum"
branch_labels = None
depends_on = None

TOOL_NAME = "get_player_transactions"
TABLES = ("chat_tools", "crm_tools")

_TAIL = (
    "NOTE: for a deposit DISPUTE, prefer get_player_latest_deposit_order "
    "instead: it targets the specific recent attempt in one call and "
    "exposes the pending/failed status detail a dispute needs."
)

# As written by 0031.
OLD_DESCRIPTION = (
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
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdraw", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdraw for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}

NEW_DESCRIPTION = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Most questions are about a "
    "deposit or a withdrawal: work out which, then filter with type "
    "(deposit or withdrawal). If the message does not make clear whether it "
    "is a deposit or a withdrawal, ask the customer one short question "
    "before calling this. Use casino or sports only when the customer "
    "clearly asks about those; omit type only for a general request "
    "like 'show my recent transactions'. " + _TAIL
)

NEW_TYPE_SPEC = {
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdrawal", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdrawal for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}


def transform_row(
    parameters, description, *, from_desc, to_desc, from_type_spec, to_type_spec,
) -> tuple[dict, str]:
    """Pure row transform: returns (new_parameters, new_description)."""
    params = dict(parameters or {})
    if params.get("type") == from_type_spec:
        new_type = dict(to_type_spec)
        new_type["enum"] = list(to_type_spec["enum"])
        params["type"] = new_type
    new_desc = to_desc if description == from_desc else description
    return params, new_desc


def _apply(from_desc, to_desc, from_type_spec, to_type_spec) -> None:
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
                from_type_spec=from_type_spec, to_type_spec=to_type_spec,
            )
            if new_params == params and new_desc == desc:
                continue
            bind.execute(
                t.update().where(t.c.id == row_id)
                .values(parameters=new_params, description=new_desc)
            )


def upgrade() -> None:
    _apply(OLD_DESCRIPTION, NEW_DESCRIPTION, OLD_TYPE_SPEC, NEW_TYPE_SPEC)


def downgrade() -> None:
    _apply(NEW_DESCRIPTION, OLD_DESCRIPTION, NEW_TYPE_SPEC, OLD_TYPE_SPEC)
