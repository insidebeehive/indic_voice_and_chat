"""Compliance config round-trips through pipeline_config (Phase 4)."""

from __future__ import annotations

from src.auth.db_resolver import tenant_context_from_row
from src.models.tenant import Tenant


def test_compliance_round_trips_via_pipeline_config():
    # Seeded under pipeline_config["compliance"]; db_resolver must pull it back into
    # TenantSettings.compliance (not the defaults).
    row = Tenant(
        id="t1", slug="t1", name="T1", status="active",
        default_language="hi", timezone="Asia/Kolkata", mode="layered",
        max_concurrent_calls=2,
        pipeline_config={
            "mode": "layered",
            "compliance": {
                "calling_hours_start": "09:00",
                "calling_hours_end": "21:00",
                "dnd_check_enabled": True,
            },
        },
    )
    row.phone_numbers = []
    row.secrets = []

    ctx = tenant_context_from_row(row)
    c = ctx.settings.compliance
    assert c.calling_hours_start == "09:00"
    assert c.calling_hours_end == "21:00"
    assert c.dnd_check_enabled is True


def test_compliance_defaults_when_absent():
    row = Tenant(
        id="t2", slug="t2", name="T2", status="active",
        default_language="hi", timezone="Asia/Kolkata", mode="layered",
        max_concurrent_calls=2, pipeline_config={"mode": "layered"})
    row.phone_numbers = []
    row.secrets = []
    # No compliance key → the default TenantCompliance (no crash).
    ctx = tenant_context_from_row(row)
    assert ctx.settings.compliance is not None
