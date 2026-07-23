"""CRM entity: crms + crm_tools tables, tenants.crm_id column, auto-seed.

Revision: 0009_crm_entity
Down: 0008_widen_chat_message_role
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0009_crm_entity"
down_revision = "0008_widen_chat_message_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "crms",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("events_webhook_url_template", sa.String(500)),
        sa.Column("auth_type", sa.String(20), server_default="api_key"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_table(
        "crm_tools",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("crm_id", sa.String(50),
                  sa.ForeignKey("crms.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("endpoint", sa.String(500), nullable=False),
        sa.Column("method", sa.String(10), server_default="GET"),
        sa.Column("parameters", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("crm_id", "name", name="uq_crm_tools_crm_name"),
    )
    op.create_index("idx_crm_tools_crm", "crm_tools", ["crm_id"])
    op.add_column(
        "tenants",
        sa.Column("crm_id", sa.String(50), sa.ForeignKey("crms.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("idx_tenants_crm", "tenants", ["crm_id"])

    _seed_betstudio_crm_and_link_tenants()


def _seed_betstudio_crm_and_link_tenants() -> None:
    """Auto-seed one 'betstudio' Crm row from today's config, seed its tool
    catalog from the hardcoded ALL_TOOLS dict, and link every existing
    tenant to it — so nothing that works today (env-var + hardcoded-catalog
    based) breaks or needs manual re-entry. events_webhook_url on tenant
    rows is intentionally left untouched (see design spec's "explicit
    override" semantics) — deriving a template is best-effort only, used
    solely to pre-fill the new Crm row for readability/future tenants."""
    from src.chatbot.catalog import ALL_TOOLS

    conn = op.get_bind()
    base_url = os.environ.get("PLATFORM_CRM_BASE_URL", "https://apistage.betstudio.io/api")
    auth_type = os.environ.get("PLATFORM_CRM_AUTH_TYPE", "api_key")

    # Best-effort webhook template: look at existing tenants' pipeline_config
    # for one with both events_webhook_url and crm.operator_id set, where the
    # URL ends with that operator_id — strip it to a {operator_id} template.
    webhook_template = None
    rows = conn.execute(sa.text(
        "SELECT pipeline_config FROM tenants WHERE pipeline_config IS NOT NULL"
    )).fetchall()
    for (pipeline_config,) in rows:
        if not pipeline_config:
            continue
        pc = pipeline_config if isinstance(pipeline_config, dict) else {}
        url = pc.get("events_webhook_url")
        operator_id = (pc.get("crm") or {}).get("operator_id")
        if url and operator_id and url.rstrip("/").endswith(operator_id):
            prefix = url.rstrip("/")[: -len(operator_id)]
            webhook_template = prefix + "{operator_id}"
            break

    conn.execute(sa.text(
        "INSERT INTO crms (id, name, base_url, events_webhook_url_template, auth_type) "
        "VALUES (:id, :name, :base_url, :webhook_template, :auth_type)"
    ), {
        "id": "betstudio", "name": "BetStudio", "base_url": base_url,
        "webhook_template": webhook_template, "auth_type": auth_type,
    })

    import json as _json
    for name, spec in ALL_TOOLS.items():
        conn.execute(sa.text(
            "INSERT INTO crm_tools (crm_id, name, description, endpoint, method, parameters) "
            "VALUES ('betstudio', :name, :description, :endpoint, :method, :parameters)"
        ), {
            "name": name, "description": spec["description"],
            "endpoint": spec["default_path"], "method": spec.get("method", "GET"),
            "parameters": _json.dumps(spec.get("parameters", {})),
        })

    conn.execute(sa.text("UPDATE tenants SET crm_id = 'betstudio' WHERE crm_id IS NULL"))


def downgrade() -> None:
    op.drop_index("idx_tenants_crm", table_name="tenants")
    op.drop_column("tenants", "crm_id")
    op.drop_index("idx_crm_tools_crm", table_name="crm_tools")
    op.drop_table("crm_tools")
    op.drop_table("crms")
