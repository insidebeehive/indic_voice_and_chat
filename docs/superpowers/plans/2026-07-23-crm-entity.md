# CRM Entity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `Crm`/`CrmTool` as first-class DB entities so CRM-level config (base URL, tool catalog, events-webhook URL template, default auth header style) is defined once per CRM and shared by every tenant registered against it, replacing today's single global env var (`PLATFORM_CRM_BASE_URL`) + hardcoded Python tool catalog (`ALL_TOOLS`).

**Architecture:** Two new tables (`crms`, `crm_tools`) plus a nullable `tenants.crm_id` FK. `resolve_crm_tools()`'s existing 3-tier precedence (tenant-registered `chat_tools` rows > CRM catalog > none) stays the same shape — tier 2 now reads from the tenant's linked `Crm` row instead of env vars + a hardcoded dict. Per-tenant auth (`crm:api_token`, `crm:x_api_key`, `operator_id`) is completely untouched. An auto-seed migration creates one `betstudio` CRM row from today's config and links every existing tenant, so nothing that works today breaks.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x (async), Alembic, pytest + pytest-asyncio, httpx (test client), vanilla JS (`static/backoffice.html`, no build step).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-23-crm-entity-design.md` — read it before starting; every task below implements one of its sections.
- "CRM" means the operator/player-data backend (BetStudio) only — never touch Chatwoot code (`chatwoot:*` secrets, `src/api/external_chat.py`).
- `crm:api_token`, `crm:x_api_key`, `operator_id` resolution is **out of scope** — do not modify how any of these three are read, stored, or encrypted.
- Every new admin-facing endpoint uses `Depends(require_admin)` from `src.auth.middleware`, matching every other platform-admin route.
- Match this repo's existing conventions exactly: SQLAlchemy 2.x `Mapped[]`/`mapped_column()` style (see `src/models/tenant.py`, `src/models/chat.py`), Alembic raw-`sa.Column` style in migration files (see `alembic/versions/0005_chat_tools.py`), FastAPI route/test patterns (see `src/api/catalog.py`, `tests/unit/test_catalog_routes.py`).
- Known pre-existing test failures, NOT to be chased in any task: `test_browser_bridge.py` (flaky), `test_campaign_loader.py::test_bharat_matka_campaign_loads_from_default_dir`, `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_config.py::test_loads_default_yaml`, `test_dev_call_control.py::test_override_set_and_pop_is_one_shot`, `test_dev_console.py` (3 tests), `test_health.py::test_health_reports_provider_names`, `test_infobip_adapter.py` (all), `test_media_storage.py` (2, missing `aiobotocore`), `test_prompts.py` (2), `test_telnyx_adapter.py` (all), `test_tenants_routes.py::test_list_tenants_shows_mode_and_models` and `::test_tenant_analytics_and_billing`.
- Run tests with `.venv/bin/python -m pytest <path> -q`.

---

### Task 1: `Crm`/`CrmTool` models + `Tenant.crm_id` column

**Files:**
- Create: `src/models/crm.py`
- Modify: `src/models/tenant.py` (add `crm_id` column to `Tenant`)
- Modify: `src/models/__init__.py` (export `Crm`, `CrmTool`)
- Test: `tests/unit/test_crm_models.py`

**Interfaces:**
- Produces: `src.models.crm.Crm` (fields: `id: str`, `name: str`, `base_url: str`, `events_webhook_url_template: Optional[str]`, `auth_type: str`, `created_at: datetime`), `src.models.crm.CrmTool` (fields: `id: int`, `crm_id: str`, `name: str`, `description: str`, `endpoint: str`, `method: str`, `parameters: dict`, `created_at: datetime`). `Tenant.crm_id: Optional[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_crm_models.py`:

```python
"""Unit tests for the Crm/CrmTool models and Tenant.crm_id."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.crm import Crm, CrmTool
from src.models.database import Base
from src.models.tenant import Tenant


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_crm_and_crm_tool_round_trip(sm):
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://apistage.betstudio.io/api",
                    events_webhook_url_template="https://bostage.betstudio.io/webhooks/crm/softphone-events/{operator_id}",
                    auth_type="api_key"))
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet",
                        description="Get wallet", endpoint="/players/{user_id}/wallet",
                        method="GET", parameters={"user_id": {"type": "string", "source": "session"}}))
        await db.commit()

    async with sm() as db:
        crm = await db.get(Crm, "betstudio")
        assert crm.base_url == "https://apistage.betstudio.io/api"
        assert crm.events_webhook_url_template.endswith("{operator_id}")
        assert crm.auth_type == "api_key"


async def test_crm_tool_unique_name_per_crm(sm):
    from sqlalchemy.exc import IntegrityError
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x", auth_type="api_key"))
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet", description="d",
                        endpoint="/e", method="GET", parameters={}))
        await db.commit()
    async with sm() as db:
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet", description="dup",
                        endpoint="/e2", method="GET", parameters={}))
        try:
            await db.commit()
            assert False, "expected IntegrityError on duplicate (crm_id, name)"
        except IntegrityError:
            pass


async def test_tenant_crm_id_nullable_and_settable(sm):
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x", auth_type="api_key"))
        db.add(Tenant(id="t1", slug="t1", name="T1"))
        await db.commit()

    async with sm() as db:
        t = await db.get(Tenant, "t1")
        assert t.crm_id is None
        t.crm_id = "betstudio"
        await db.commit()

    async with sm() as db:
        t = await db.get(Tenant, "t1")
        assert t.crm_id == "betstudio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.crm'`

- [ ] **Step 3: Write the model file**

Create `src/models/crm.py`:

```python
"""CRM entity: the downstream operator/player-data backend (e.g. BetStudio).

A ``Crm`` row is CRM-level config shared by every tenant registered against
it (``Tenant.crm_id``): the base URL its tool endpoints are joined onto, the
webhook URL template events are POSTed to (with ``{operator_id}`` substituted
from the tenant's own operator_id at send time), and the default auth header
style. Genuinely per-tenant/operator values — the auth token/x-api-key,
operator_id itself — stay on the tenant (``tenant_secrets`` / pipeline_config),
completely untouched by this entity.

``CrmTool`` rows are this CRM's tool catalog (replaces the old hardcoded
``src.chatbot.catalog.ALL_TOOLS`` dict) — one row per tool, DB-backed so a
new CRM's tools can be registered without a code deploy.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
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


class Crm(Base):
    __tablename__ = "crms"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    events_webhook_url_template: Mapped[str | None] = mapped_column(String(500))
    auth_type: Mapped[str] = mapped_column(String(20), default="api_key")  # api_key|bearer
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())


class CrmTool(Base):
    __tablename__ = "crm_tools"
    __table_args__ = (UniqueConstraint("crm_id", "name", name="uq_crm_tools_crm_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crm_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    method: Mapped[str] = mapped_column(String(10), default="GET")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
```

Modify `src/models/tenant.py`: add the import and column. In the imports block near the top, add `ForeignKey` is already imported — no import change needed. In the `Tenant` class, add this column right after `pipeline_config` (before `created_at`):

```python
    crm_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="SET NULL"), nullable=True, index=True
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Export from `src/models/__init__.py`**

Modify `src/models/__init__.py` — add `Crm`, `CrmTool` to the import from `src.models.crm` and to `__all__`:

```python
from src.models.benchmark import BenchmarkRun, KBDocument, PlatformKBDocument
from src.models.campaign import Campaign, Lead
from src.models.chat import ChatMessage, ChatSession, ChatTool
from src.models.conversation import Conversation, Event, Turn
from src.models.crm import Crm, CrmTool
from src.models.database import Base, get_engine, get_sessionmaker
from src.models.tenant import (
    ProviderCost,
    Tenant,
    TenantApiKey,
    TenantPhoneNumber,
    TenantSecret,
)

__all__ = [
    "Base",
    "BenchmarkRun",
    "Campaign",
    "ChatMessage",
    "ChatSession",
    "ChatTool",
    "Conversation",
    "Crm",
    "CrmTool",
    "Event",
    "KBDocument",
    "PlatformKBDocument",
    "Lead",
    "ProviderCost",
    "Tenant",
    "TenantApiKey",
    "TenantPhoneNumber",
    "TenantSecret",
    "Turn",
    "get_engine",
    "get_sessionmaker",
]
```

- [ ] **Step 6: Run the full test file once more + a broader sanity sweep**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_models.py tests/unit/test_tenants_routes.py -q`
Expected: all pass except the 2 known pre-existing `test_tenants_routes.py` failures listed in Global Constraints (unrelated to this change — the new nullable `crm_id` column must not affect existing tenant construction anywhere, since it's optional and defaults to `None`).

- [ ] **Step 7: Commit**

```bash
git add src/models/crm.py src/models/tenant.py src/models/__init__.py tests/unit/test_crm_models.py
git commit -m "feat(crm): add Crm/CrmTool models and Tenant.crm_id column"
```

---

### Task 2: Alembic migration — schema + auto-seed

**Files:**
- Create: `alembic/versions/0009_crm_entity.py`
- Test: manual verification (this repo's test suite uses in-memory SQLite via `Base.metadata.create_all`, not Alembic migrations directly — there is no existing pattern for unit-testing a migration file's SQL. Verify by syntax-checking the file and hand-tracing against Task 1's models, then running it against a real reachable Postgres as described in Step 5.)

**Interfaces:**
- Consumes: `Crm`, `CrmTool` from Task 1 (the migration's `upgrade()` creates tables matching those models exactly — table/column names must match byte-for-byte).
- Produces: `crms` table with one seeded row (`id='betstudio'`), `crm_tools` table with 18 seeded rows (one per `src.chatbot.catalog.ALL_TOOLS` entry), every pre-existing tenant's `crm_id` set to `'betstudio'`.

- [ ] **Step 1: Write the migration file**

Create `alembic/versions/0009_crm_entity.py`:

```python
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
```

- [ ] **Step 2: Check the migration chains correctly onto the current head**

Run: `grep -n "down_revision" alembic/versions/0008_widen_chat_message_role.py`
Expected output confirms `0008_widen_chat_message_role` has no migration after it (i.e. it's the current head) — if some other migration already declares `down_revision = "0008_widen_chat_message_role"`, STOP and re-chain this migration onto the actual current head instead (`ls alembic/versions/` and re-check).

- [ ] **Step 3: Syntax-check the file**

Run: `.venv/bin/python -c "import ast; ast.parse(open('alembic/versions/0009_crm_entity.py').read())"`
Expected: no output (parses cleanly).

- [ ] **Step 4: Dry-run against a disposable local SQLite copy to catch gross errors**

Alembic's `env.py` in this repo is Postgres-oriented (schema handling), so a full `alembic upgrade head` dry run against SQLite isn't representative — instead, hand-verify by running just the seed helper's core logic against an in-memory SQLite DB matching the new schema, reusing Task 1's fixture pattern:

```bash
.venv/bin/python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text
from src.models.database import Base

async def main():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:', future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        from src.chatbot.catalog import ALL_TOOLS
        print('ALL_TOOLS has', len(ALL_TOOLS), 'entries — migration will insert this many crm_tools rows')
    await engine.dispose()

asyncio.run(main())
"
```
Expected: prints `ALL_TOOLS has 18 entries — migration will insert this many crm_tools rows` (or whatever the current count is) — confirms the import used by the migration resolves and the seed loop has a real, non-empty source to iterate.

- [ ] **Step 5: Run the real migration against the actual reachable database**

This is the real verification (per the note above, SQLite can't stand in for the schema-aware Postgres path this migration exercises). Run against whichever database `stage`'s deployment currently uses (the one already confirmed reachable and containing the `stage` tenant's real data from this session's earlier debugging):

```bash
alembic upgrade head
```
Expected: no errors; then confirm:
```bash
.venv/bin/python -c "
import asyncio
from sqlalchemy import text
from src.models.database import get_engine

async def main():
    engine = get_engine()
    async with engine.begin() as conn:
        r = await conn.execute(text('SELECT id, name, base_url FROM voicebot.crms'))
        print('crms:', r.fetchall())
        r = await conn.execute(text('SELECT count(*) FROM voicebot.crm_tools'))
        print('crm_tools count:', r.scalar())
        r = await conn.execute(text(\"SELECT slug, crm_id FROM voicebot.tenants\"))
        print('tenant links:', r.fetchall())

asyncio.run(main())
"
```
Expected: one `betstudio` CRM row, 18 `crm_tools` rows, and every existing tenant (including `stage`) shows `crm_id = 'betstudio'`.

- [ ] **Step 6: Commit**

```bash
git add alembic/versions/0009_crm_entity.py
git commit -m "feat(crm): migration — crms/crm_tools tables, auto-seed from current config"
```

---

### Task 3: `resolve_crm_tools()` reads from the linked `Crm`

**Files:**
- Modify: `src/bootstrap.py:228-338` (the `resolve_crm_tools` function)
- Test: `tests/unit/test_crm_tools_platform_fallback.py` (extend existing file)

**Interfaces:**
- Consumes: `Tenant.crm_id` (Task 1), `Crm`/`CrmTool` models (Task 1). `tenant.settings.crm.operator_id` (existing, untouched) and `tenant.secrets_resolved.get("crm:api_token")`/`.get("crm:x_api_key")` (existing, untouched).
- Produces: same `resolve_crm_tools(tenant, sessionmaker) -> tuple[list[ToolSpec], dict[str, dict], str]` signature as today — `source` now returns `"crm_catalog"` (renamed from `"platform_fallback"`) for tier 2.

- [ ] **Step 1: Read the current function in full**

Run: `grep -n "async def resolve_crm_tools" -A 110 src/bootstrap.py`

This shows the exact current tenant-rows branch (keep byte-for-byte unchanged) and the platform-fallback branch (this is what gets replaced).

- [ ] **Step 2: Write the failing test**

Add to `tests/unit/test_crm_tools_platform_fallback.py` (this file already has fixtures for tenants with/without CRM secrets from earlier work — follow its existing pattern for constructing a `TenantContext`/`TenantSettings` and an in-memory sessionmaker):

```python
async def test_crm_linked_tenant_gets_crm_catalog_tools(sm_with_crm_seed):
    """A tenant with tenant.crm_id set gets that CRM's DB-backed tool catalog,
    using the CRM's base_url/auth_type and the tenant's OWN api_token/x_api_key
    (unchanged per-tenant resolution)."""
    from src.bootstrap import resolve_crm_tools
    from src.config_tenant import TenantSettings, TenantPipelineConfig, TenantCRMConfig
    from src.auth.context import TenantContext

    tenant = TenantContext(
        settings=TenantSettings(
            id="t1", slug="t1", name="T1", crm_id="betstudio",
            pipeline=TenantPipelineConfig(crm=TenantCRMConfig(operator_id="op-123")),
        ),
        secrets_resolved={"crm:api_token": "tok-abc", "crm:x_api_key": "key-xyz"},
    )

    specs, execs, source = await resolve_crm_tools(tenant, sm_with_crm_seed)

    assert source == "crm_catalog"
    assert len(specs) == 18  # matches the seeded catalog's tool count
    sample = execs["get_player_wallet"]
    assert sample["endpoint"] == "https://apistage.betstudio.io/api/players/{user_id}/wallet"
    assert sample["auth_type"] == "api_key"
    assert sample["token"] == "tok-abc"
    assert sample["x_api_key"] == "key-xyz"
    assert sample["extra_headers"] == {"operatorid": "op-123"}


async def test_tenant_without_crm_link_and_no_chat_tools_gets_none(sm_with_crm_seed):
    from src.bootstrap import resolve_crm_tools
    from src.config_tenant import TenantSettings
    from src.auth.context import TenantContext

    tenant = TenantContext(settings=TenantSettings(id="t2", slug="t2", name="T2"), secrets_resolved={})

    specs, execs, source = await resolve_crm_tools(tenant, sm_with_crm_seed)
    assert source == "none"
    assert specs == []
```

Before writing these, check `src/auth/context.py` for `TenantContext`'s exact constructor and `.id` property (it may derive `.id` from `settings.id` automatically rather than needing manual assignment) — adjust the test's tenant-construction lines to match whatever this repo's existing tests in the same file already do (they already build `TenantContext` instances for the pre-CRM-entity tests in this file — copy that exact pattern rather than the sketch above).

Add a new fixture `sm_with_crm_seed` near the top of the file (or reuse/extend the file's existing sessionmaker fixture) that also seeds one `Crm` + its `CrmTool` rows:

```python
@pytest_asyncio.fixture
async def sm_with_crm_seed():
    from src.models.crm import Crm, CrmTool
    from src.chatbot.catalog import ALL_TOOLS

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio",
                    base_url="https://apistage.betstudio.io/api", auth_type="api_key"))
        for name, spec in ALL_TOOLS.items():
            db.add(CrmTool(crm_id="betstudio", name=name, description=spec["description"],
                            endpoint=spec["default_path"], method=spec.get("method", "GET"),
                            parameters=spec.get("parameters", {})))
        await db.commit()
    yield sm
    await engine.dispose()
```
(Match existing imports in the file — `create_async_engine`, `async_sessionmaker`, `Base` should already be imported there.)

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_tools_platform_fallback.py -k crm_linked -v`
Expected: FAIL — either an `AttributeError` (no `tenant.crm_id` handling yet) or `source == "none"`/`"platform_fallback"` instead of `"crm_catalog"`.

- [ ] **Step 4: Rewrite the platform-fallback branch**

In `src/bootstrap.py`, replace the entire block from `# ── Platform catalog fallback ─────` through the `return specs, execs, "platform_fallback"` line with:

```python
    # ── CRM catalog (tenant linked to a Crm entity) ─────────────────────
    crm_id = getattr(tenant, "crm_id", None) or getattr(tenant.settings, "crm_id", None)
    if not crm_id:
        return [], {}, "none"

    from src.models.crm import Crm, CrmTool

    async with sessionmaker() as db:
        crm = await db.get(Crm, crm_id)
        if crm is None:
            return [], {}, "none"
        crm_tool_rows = (await db.execute(
            select(CrmTool).where(CrmTool.crm_id == crm_id)
        )).scalars().all()

    if not crm_tool_rows:
        return [], {}, "none"

    api_token = sr.get("crm:api_token")
    x_api_key = sr.get("crm:x_api_key")
    operator_id = getattr(tenant.settings.crm, "operator_id", None) or tenant.id
    extra_headers = {"operatorid": operator_id}

    for row in crm_tool_rows:
        endpoint = crm.base_url.rstrip("/") + row.endpoint
        specs.append(ToolSpec(
            name=row.name, description=row.description,
            parameters=_crm_params_to_schema(row.parameters)))
        execs[row.name] = {
            "endpoint": endpoint, "method": row.method,
            "parameters": row.parameters or {}, "auth_type": crm.auth_type,
            "token": api_token, "x_api_key": x_api_key,
            "extra_headers": extra_headers,
        }
    return specs, execs, "crm_catalog"
```

Confirm `select` is already imported at module/function level in `resolve_crm_tools` (it is — the tenant-rows branch above already uses it). Do NOT touch the tenant-registered-tools branch above this (`if specs: return specs, execs, "tenant"`) — leave it completely as-is.

**`crm_id` must be threaded through explicitly — confirmed it isn't today.** `tenant_context_from_row` (`src/auth/db_resolver.py:32-66`) builds `TenantSettings(...)` field-by-field from the `Tenant` ORM row (e.g. `id=tenant.id, slug=tenant.slug, ..., crm=crm, ...`) — it does NOT pass through arbitrary row attributes, so `crm_id` (Task 1's new column) needs two explicit additions:

1. In `src/config_tenant.py`'s `TenantSettings` class, add `crm_id: Optional[str] = None`.
2. In `src/auth/db_resolver.py`'s `tenant_context_from_row`, add `crm_id=tenant.crm_id,` to the `TenantSettings(...)` constructor call (alongside the existing `id=tenant.id` etc. lines).

Then, in the `resolve_crm_tools` snippet above, use `tenant.settings.crm_id` exactly (not `tenant.crm_id` — `TenantContext` does not expose it as a direct attribute; it lives on `.settings`). Replace the line
```python
    crm_id = getattr(tenant, "crm_id", None) or getattr(tenant.settings, "crm_id", None)
```
with:
```python
    crm_id = tenant.settings.crm_id
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_tools_platform_fallback.py -v`
Expected: all pass, including the pre-existing tests in this file (tenant-secret-wins-over-crm-default, tenant-registered-tools-precedence, x_api_key population — these should all still pass since only the fallback branch's SOURCE of base_url/auth_type/tools changed, not the token/x_api_key/extra_headers resolution shape).

- [ ] **Step 6: Run the broader suite**

Run: `.venv/bin/python -m pytest tests/unit -q -k "bootstrap or crm_tools or chat_tools"`
Expected: all pass except known pre-existing failures (see Global Constraints).

- [ ] **Step 7: Commit**

```bash
git add src/bootstrap.py src/auth/db_resolver.py src/config_tenant.py tests/unit/test_crm_tools_platform_fallback.py
git commit -m "feat(crm): resolve_crm_tools reads the tenant's linked Crm catalog"
```

---

### Task 4: `events_webhook_url` template resolution

**Files:**
- Create: helper function in `src/integration/tenant_events.py` (or a new small module `src/integration/crm_webhook_url.py` if `tenant_events.py` is already large — check its current line count first with `wc -l src/integration/tenant_events.py`; if under ~200 lines, add the helper there instead of a new file)
- Modify: `src/main.py` (the `_notify_tenant_event`/`_resolve_tenant_event_secret` area, ~line 296-309)
- Modify: `src/api/chat_webhooks.py` (`send_bo_webhook`, ~line 22-42)
- Test: `tests/unit/test_tenant_events.py` (extend)

**Interfaces:**
- Produces: `resolve_events_webhook_url(tenant: TenantContext) -> Optional[str]` — returns the tenant's explicit `events_webhook_url` override if set, else the tenant's linked CRM's `events_webhook_url_template` with `{operator_id}` substituted, else `None`.
- Consumes: `tenant.settings.events_webhook_url` (existing), `tenant.settings.crm_id`/`Crm.events_webhook_url_template` (Task 1/3), `tenant.settings.crm.operator_id` (existing).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_tenant_events.py`:

```python
async def test_resolve_events_webhook_url_uses_tenant_override_when_set():
    from src.integration.tenant_events import resolve_events_webhook_url
    from src.config_tenant import TenantSettings
    from src.auth.context import TenantContext

    tenant = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1", events_webhook_url="https://explicit.example.com/hook"))
    url = await resolve_events_webhook_url(tenant, sessionmaker=None)
    assert url == "https://explicit.example.com/hook"


async def test_resolve_events_webhook_url_uses_crm_template_with_operator_id(sm_with_crm_seed):
    from src.integration.tenant_events import resolve_events_webhook_url
    from src.config_tenant import TenantSettings, TenantPipelineConfig, TenantCRMConfig
    from src.auth.context import TenantContext
    from src.models.crm import Crm

    async with sm_with_crm_seed() as db:
        crm = await db.get(Crm, "betstudio")
        crm.events_webhook_url_template = "https://bostage.betstudio.io/webhooks/crm/softphone-events/{operator_id}"
        await db.commit()

    tenant = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1", crm_id="betstudio",
        pipeline=TenantPipelineConfig(crm=TenantCRMConfig(operator_id="ab858a8c-7ad4-47d2-a0b7-05ee93f8f134"))))
    url = await resolve_events_webhook_url(tenant, sm_with_crm_seed)
    assert url == "https://bostage.betstudio.io/webhooks/crm/softphone-events/ab858a8c-7ad4-47d2-a0b7-05ee93f8f134"


async def test_resolve_events_webhook_url_none_when_nothing_configured():
    from src.integration.tenant_events import resolve_events_webhook_url
    from src.config_tenant import TenantSettings
    from src.auth.context import TenantContext

    tenant = TenantContext(settings=TenantSettings(id="t1", slug="t1", name="T1"))
    url = await resolve_events_webhook_url(tenant, sessionmaker=None)
    assert url is None
```

Adjust `TenantContext`/`TenantSettings` construction to match this file's existing patterns exactly (check the top of `tests/unit/test_tenant_events.py`, which already builds these objects for the earlier webhook-secret tests from this session).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_tenant_events.py -k resolve_events_webhook_url -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_events_webhook_url'`

- [ ] **Step 3: Implement the helper**

In `src/integration/tenant_events.py`, add:

```python
async def resolve_events_webhook_url(tenant, sessionmaker) -> "str | None":
    """The URL to POST tenant lifecycle events to.

    Priority: the tenant's own explicit ``events_webhook_url`` (an escape
    hatch for a tenant needing a different shape than its CRM's default),
    else the tenant's linked CRM's ``events_webhook_url_template`` with
    ``{operator_id}`` substituted from the tenant's own operator_id, else
    None (no webhook configured at all)."""
    explicit = getattr(tenant.settings, "events_webhook_url", None)
    if explicit:
        return explicit

    crm_id = getattr(tenant.settings, "crm_id", None)
    if not crm_id or sessionmaker is None:
        return None

    from src.models.crm import Crm

    async with sessionmaker() as db:
        crm = await db.get(Crm, crm_id)
    if crm is None or not crm.events_webhook_url_template:
        return None

    operator_id = getattr(tenant.settings.crm, "operator_id", None) or tenant.id
    return crm.events_webhook_url_template.replace("{operator_id}", operator_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_tenant_events.py -v`
Expected: all pass, including pre-existing tests in this file.

- [ ] **Step 5: Wire into `src/main.py`'s `_notify_tenant_event`**

Read the current code first: `grep -n "events_webhook_url\|def _notify_tenant_event" -A 5 src/main.py`. Replace the line `url = getattr(settings, "events_webhook_url", None)` with a call to the new helper, passing whatever `sessionmaker`/`tenant` are already in scope at that point in the function (check exact variable names in context before editing — this closure already has access to the app's sessionmaker per earlier work this session on `_resolve_tenant_event_secret`, follow that same pattern for obtaining it):

```python
        url = await resolve_events_webhook_url(tenant, sessionmaker)
```
Add the import at the top of `src/main.py`: `from src.integration.tenant_events import resolve_events_webhook_url` (alongside whatever it already imports from that module for `deliver_tenant_event`/`_resolve_tenant_event_secret`).

- [ ] **Step 6: Wire into `src/api/chat_webhooks.py`'s `send_bo_webhook`**

Same treatment: replace `url = getattr(settings, "events_webhook_url", None)` with `url = await resolve_events_webhook_url(tenant, get_sessionmaker())` (import `get_sessionmaker` from `src.models.database` if not already imported in this file — check first) and `from src.integration.tenant_events import resolve_events_webhook_url` at the top.

- [ ] **Step 7: Run the full webhook test suite**

Run: `.venv/bin/python -m pytest tests/unit -q -k "webhook or tenant_event or chat_webhooks"`
Expected: all pass, no regressions to the existing "unsigned warning" tests from earlier this session.

- [ ] **Step 8: Commit**

```bash
git add src/integration/tenant_events.py src/main.py src/api/chat_webhooks.py tests/unit/test_tenant_events.py
git commit -m "feat(crm): resolve events_webhook_url from the tenant's CRM template"
```

---

### Task 5: CRM CRUD API

**Files:**
- Create: `src/api/crms.py`
- Modify: `src/api/__init__.py` (register the router)
- Test: `tests/unit/test_crm_routes.py`

**Interfaces:**
- Produces: `GET /api/v1/crms` (list), `POST /api/v1/crms` (create), `GET /api/v1/crms/{id}` (detail incl. tools), `PATCH /api/v1/crms/{id}` (update fields and/or replace tool list). All admin-gated via `Depends(require_admin)`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_crm_routes.py`, mirroring `tests/unit/test_catalog_routes.py`'s fixture pattern exactly:

```python
"""Route tests for CRM entity CRUD (admin-only)."""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import crms
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.database import Base

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])
    app = FastAPI()
    app.include_router(crms.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_admin_tokens([])
    await engine.dispose()


async def test_create_and_get_crm(client: AsyncClient) -> None:
    resp = await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://apistage.betstudio.io/api",
        "auth_type": "api_key",
        "tools": [{"name": "get_player_wallet", "description": "d",
                   "endpoint": "/players/{user_id}/wallet", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text

    resp = await client.get("/crms/betstudio", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == "https://apistage.betstudio.io/api"
    assert len(body["tools"]) == 1


async def test_list_crms_requires_admin(client: AsyncClient) -> None:
    assert (await client.get("/crms")).status_code == 401


async def test_patch_crm_replaces_tool_list(client: AsyncClient) -> None:
    await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://x",
        "tools": [{"name": "a", "description": "d", "endpoint": "/a", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)

    resp = await client.patch("/crms/betstudio", json={
        "tools": [{"name": "b", "description": "d2", "endpoint": "/b", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200

    detail = (await client.get("/crms/betstudio", headers=ADMIN_HEADERS)).json()
    names = {t["name"] for t in detail["tools"]}
    assert names == {"b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.crms'`

- [ ] **Step 3: Write the router**

Create `src/api/crms.py`:

```python
"""Admin CRUD for Crm entities + their tool catalogs.

- ``GET    /crms``       list every CRM (id, name, base_url)
- ``POST   /crms``       create a CRM + its initial tool list
- ``GET    /crms/{id}``  detail including the full tool list
- ``PATCH  /crms/{id}``  update CRM-level fields and/or replace the tool list

Admin-only (``require_admin``) — a CRM is shared platform config, not
tenant-scoped. A tenant is linked to a CRM via ``PATCH /tenants/{id}``
(``crm_id`` field), not through this router.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth.middleware import require_admin
from src.models.crm import Crm, CrmTool

router = APIRouter(prefix="/crms", tags=["crms"])


class CrmToolIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    method: str = "GET"
    parameters: dict = Field(default_factory=dict)


class CrmToolOut(CrmToolIn):
    id: int


class CrmCreateRequest(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    events_webhook_url_template: Optional[str] = None
    auth_type: str = "api_key"
    tools: list[CrmToolIn] = Field(default_factory=list)


class CrmUpdateRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    events_webhook_url_template: Optional[str] = None
    auth_type: Optional[str] = None
    tools: Optional[list[CrmToolIn]] = None  # if provided, REPLACES the entire tool list


class CrmSummary(BaseModel):
    id: str
    name: str
    base_url: str


class CrmDetail(CrmSummary):
    events_webhook_url_template: Optional[str]
    auth_type: str
    tools: list[CrmToolOut]


@router.get("", response_model=list[CrmSummary])
async def list_crms(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> list[CrmSummary]:
    rows = (await session.execute(select(Crm))).scalars().all()
    return [CrmSummary(id=r.id, name=r.name, base_url=r.base_url) for r in rows]


@router.post("", response_model=CrmDetail)
async def create_crm(
    req: CrmCreateRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    existing = await session.get(Crm, req.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"CRM {req.id!r} already exists")
    crm = Crm(id=req.id, name=req.name, base_url=req.base_url,
               events_webhook_url_template=req.events_webhook_url_template,
               auth_type=req.auth_type)
    session.add(crm)
    for t in req.tools:
        session.add(CrmTool(crm_id=req.id, name=t.name, description=t.description,
                             endpoint=t.endpoint, method=t.method, parameters=t.parameters))
    await session.commit()
    return await _detail(session, req.id)


@router.get("/{crm_id}", response_model=CrmDetail)
async def get_crm(
    crm_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    return await _detail(session, crm_id)


@router.patch("/{crm_id}", response_model=CrmDetail)
async def update_crm(
    crm_id: str,
    req: CrmUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    crm = await session.get(Crm, crm_id)
    if crm is None:
        raise HTTPException(status_code=404, detail="CRM not found")
    if req.name is not None:
        crm.name = req.name
    if req.base_url is not None:
        crm.base_url = req.base_url
    if req.events_webhook_url_template is not None:
        crm.events_webhook_url_template = req.events_webhook_url_template
    if req.auth_type is not None:
        crm.auth_type = req.auth_type
    if req.tools is not None:
        existing_tools = (await session.execute(
            select(CrmTool).where(CrmTool.crm_id == crm_id)
        )).scalars().all()
        for t in existing_tools:
            await session.delete(t)
        await session.flush()
        for t in req.tools:
            session.add(CrmTool(crm_id=crm_id, name=t.name, description=t.description,
                                 endpoint=t.endpoint, method=t.method, parameters=t.parameters))
    await session.commit()
    return await _detail(session, crm_id)


async def _detail(session: AsyncSession, crm_id: str) -> CrmDetail:
    crm = await session.get(Crm, crm_id)
    if crm is None:
        raise HTTPException(status_code=404, detail="CRM not found")
    tools = (await session.execute(
        select(CrmTool).where(CrmTool.crm_id == crm_id)
    )).scalars().all()
    return CrmDetail(
        id=crm.id, name=crm.name, base_url=crm.base_url,
        events_webhook_url_template=crm.events_webhook_url_template,
        auth_type=crm.auth_type,
        tools=[CrmToolOut(id=t.id, name=t.name, description=t.description,
                           endpoint=t.endpoint, method=t.method, parameters=t.parameters)
               for t in tools],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_crm_routes.py -v`
Expected: 3 passed

- [ ] **Step 5: Register the router**

Modify `src/api/__init__.py` — add `crms` to the import tuple (alphabetically, between `config_routes` and `conversations`... actually alphabetically `crms` sits between `config_routes` and `conversations`, verify exact placement) and add `api_router.include_router(crms.router)`:

```python
from src.api import (
    benchmarks,
    calls,
    campaigns,
    catalog,
    chat,
    chat_tools,
    config_routes,
    conversations,
    crms,
    external_chat,
    knowledge,
    sessions,
    softphone,
    telephony_crm,
    telephony_hooks,
    tenants,
    webhooks_routes,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions.router)
api_router.include_router(tenants.router)
api_router.include_router(catalog.router)
api_router.include_router(campaigns.router)
api_router.include_router(calls.router)
api_router.include_router(softphone.router)
api_router.include_router(config_routes.router)
api_router.include_router(conversations.router)
api_router.include_router(crms.router)
api_router.include_router(knowledge.router)
api_router.include_router(webhooks_routes.router)
api_router.include_router(benchmarks.router)
api_router.include_router(chat.router)
api_router.include_router(chat_tools.router)
api_router.include_router(external_chat.router)
api_router.include_router(telephony_hooks.router)
api_router.include_router(telephony_crm.router)
```

(Keep every existing line — only the `crms` import and its one new `include_router` call are additions.)

- [ ] **Step 6: Run the broader route-registration sanity check**

Run: `.venv/bin/python -c "from src.api import api_router; print([r.path for r in api_router.routes if 'crms' in r.path])"`
Expected: prints the 4 new route paths (`/api/v1/crms`, `/api/v1/crms/{crm_id}` twice for GET/PATCH, etc.)

- [ ] **Step 7: Commit**

```bash
git add src/api/crms.py src/api/__init__.py tests/unit/test_crm_routes.py
git commit -m "feat(crm): admin CRUD API for Crm entities"
```

---

### Task 6: Relabel `/chat/tools/resolved` + surface `crm_id`

**Files:**
- Modify: `src/api/chat_tools.py` (the `ResolvedToolsResponse`/`list_resolved_tools` area)
- Test: `tests/unit/test_chat_tools_resolved_endpoint.py` (extend/update existing tests)

**Interfaces:**
- Consumes: the `"crm_catalog"` source string from Task 3's `resolve_crm_tools`.
- Produces: `ResolvedToolsResponse.source` now includes `"crm_catalog"` as a possible value (was `"platform_fallback"`); adds `crm_id: Optional[str]` field.

- [ ] **Step 1: Update the existing tests to expect the new source name**

In `tests/unit/test_chat_tools_resolved_endpoint.py`, find every assertion of `source == "platform_fallback"` (grep `grep -n "platform_fallback" tests/unit/test_chat_tools_resolved_endpoint.py`) and change to `"crm_catalog"`. Add one new assertion to the existing CRM-fallback test: `assert body["crm_id"] == "betstudio"` (or whatever CRM id the test's fixture seeds — check Task 3's fixture naming).

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_chat_tools_resolved_endpoint.py -v`
Expected: FAIL — `source` still says the old value / `crm_id` field doesn't exist yet on the response model.

- [ ] **Step 3: Update the response model + endpoint**

In `src/api/chat_tools.py`, find `ResolvedToolsResponse` and add `crm_id: Optional[str] = None`. In `list_resolved_tools`, after calling `resolve_crm_tools(...)`, determine `crm_id` — the simplest correct source is `getattr(tenant.settings, "crm_id", None)` (only meaningful when `source == "crm_catalog"`; leave `None` otherwise) and pass it into the response construction.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_chat_tools_resolved_endpoint.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/api/chat_tools.py tests/unit/test_chat_tools_resolved_endpoint.py
git commit -m "feat(crm): relabel platform_fallback -> crm_catalog, surface crm_id"
```

---

### Task 7: Backoffice — CRM dropdown replaces Base URL/Auth type on tenant edit

**Files:**
- Modify: `static/backoffice.html` (`loadCRM`/`saveCrmConfig` functions)

**Interfaces:**
- Consumes: `GET /api/v1/crms` (Task 5) for the dropdown options; `PATCH /api/v1/tenants/{id}` gains a `crm_id` field (verify/add this to `UpdateTenantRequest` in `src/api/tenants.py` as part of this task if not already covered by Task 3's `TenantSettings.crm_id` change — check whether the tenant update route already forwards arbitrary `pipeline_config` fields or needs an explicit `crm_id` request field added).

- [ ] **Step 1: Add `crm_id` to the tenant update request/response, if not already wired**

Check `src/api/tenants.py`'s `UpdateTenantRequest`/`update_tenant` (grep `grep -n "class UpdateTenantRequest" -A 20 src/api/tenants.py`). Add `crm_id: Optional[str] = None` to the request model, and in the handler, when `req.crm_id is not None`, set `t.crm_id = req.crm_id` directly on the `Tenant` row (this is a real column from Task 1, not a `pipeline_config` JSON key — assign it directly). Add `crm_id` to whatever response model surfaces tenant detail (`TenantSummary` or similar) so the backoffice can read the currently-linked CRM back.

- [ ] **Step 2: Update `loadCRM` in `static/backoffice.html`**

Read the current function in full first: `grep -n "async function loadCRM" -A 40 static/backoffice.html`.

Replace the Base URL / Auth type input fields with a CRM dropdown, fetched from the new endpoint. Modify `loadCRM(id, slug)`:

```javascript
async function loadCRM(id, slug) {
  const [cr, crmsResp] = await Promise.all([
    apiSend("GET", `/api/v1/tenants/${id}/chat-config`),
    apiSend("GET", "/api/v1/crms"),
  ]);
  const cfg    = cr.ok ? cr.json : {};
  const crmCfg = cfg.crm || {};
  const crms   = crmsResp.ok ? crmsResp.json : [];
  const currentCrmId = (window.TENANTS || {})[id]?.crm_id || "";

  const crmOptions = crms.map(c =>
    `<option value="${escAttr(c.id)}" ${c.id === currentCrmId ? "selected" : ""}>${escT(c.name)}</option>`
  ).join("");

  $("pane_crm").innerHTML = `
    <div class="hint" style="margin-bottom:.6rem;">
      CRM tools (wallet, bets, transactions, bonuses, profile, operator config) come
      from the CRM this tenant is registered against. Base URL and the tool catalog
      are managed at the CRM level (see the CRMs admin page), not per tenant.
    </div>
    <h2>CRM</h2>
    <div class="creds" style="gap:.6rem;flex-wrap:wrap;">
      <label>CRM
        <select id="ch_crm_id">
          <option value="">— none —</option>
          ${crmOptions}
        </select>
      </label>
    </div>
    <div style="margin-top:.5rem;">
      <button onclick="saveCrmConfig('${escAttr(slug)}')">Save CRM link</button>
      <span id="ch_result" class="hint" style="margin-left:.6rem;"></span>
    </div>
    <h2 style="margin-top:1.2rem;">CRM auth (tenant-specific)</h2>
    <div class="hint" style="margin-bottom:.5rem;">
      This tenant's own credentials for the linked CRM — different tenants on the
      same CRM each need their own here.
    </div>
    <div class="creds" style="gap:.6rem;flex-wrap:wrap;">
      <label>API token <small class="hint">(encrypted at rest)</small>
        <input id="ch_token" type="password" style="width:18rem" placeholder="token value" />
      </label>
      <label>X-API-Key <small class="hint">(encrypted at rest, sent alongside Authorization)</small>
        <input id="ch_xapikey" type="password" style="width:18rem" placeholder="key value" />
      </label>
      <label>Operator ID
        <input id="ch_opid" style="width:16rem" placeholder="ab858a8c-7ad4-47d2-a0b7-…" />
      </label>
    </div>
    <div style="margin-top:.8rem;">
      <button onclick="saveCrmAuth('${escAttr(slug)}')">Save CRM auth</button>
      <span id="ch_auth_result" class="hint" style="margin-left:.6rem;"></span>
    </div>`;

  if (crmCfg.api_token)   $("ch_token").placeholder   = "... (already set — leave blank to keep)";
  if (crmCfg.x_api_key)   $("ch_xapikey").placeholder = "... (already set — leave blank to keep)";
  if (crmCfg.operator_id) $("ch_opid").value          = crmCfg.operator_id;
}
```

Note: this splits the old single `saveCrmConfig` into two actions — linking the CRM (`crm_id`, a `Tenant` column, via `PATCH /tenants/{id}`) and saving the tenant's own auth (`api_token`/`x_api_key`/`operator_id`, still the existing `crm` block in the same `PATCH /tenants/{id}` body). Base URL and auth_type are gone from this form entirely — they no longer belong here.

- [ ] **Step 3: Update `saveCrmConfig` and add `saveCrmAuth`**

Replace the existing `saveCrmConfig` function:

```javascript
async function saveCrmConfig(slug) {
  const crm_id = $("ch_crm_id").value || null;
  $("ch_result").textContent = "saving…";
  const tid = SELECTED;
  const r = await apiSend("PATCH", `/api/v1/tenants/${tid}`, { crm_id });
  $("ch_result").textContent = r.ok ? "✓ CRM link saved" : `Error ${r.status}: ${r.json.detail || JSON.stringify(r.json)}`;
}

async function saveCrmAuth(slug) {
  const api_token   = $("ch_token").value.trim();
  const x_api_key   = $("ch_xapikey").value.trim();
  const operator_id = $("ch_opid").value.trim();
  const crm = {};
  if (api_token)   crm.api_token   = api_token;
  if (x_api_key)   crm.x_api_key   = x_api_key;
  if (operator_id) crm.operator_id = operator_id;
  $("ch_auth_result").textContent = "saving…";
  const tid = SELECTED;
  const r = await apiSend("PATCH", `/api/v1/tenants/${tid}`, { crm });
  $("ch_auth_result").textContent = r.ok ? "✓ CRM auth saved" : `Error ${r.status}: ${r.json.detail || JSON.stringify(r.json)}`;
}
```

(Verify the exact request body shape the backend already expects for `x_api_key` — it was added earlier this session; confirm the field name matches exactly what `src/api/tenants.py`'s CRM config model calls it before shipping this.)

- [ ] **Step 4: Manual verification (no JS test harness in this repo)**

If `node` is available: `node --check` (or extract the `<script>` block and run `node -e "new Function(...)"`) to confirm no syntax errors, matching how this file's edits were verified earlier this session. Re-read the full edited `loadCRM`/`saveCrmConfig`/`saveCrmAuth` block once more for quote-escaping correctness (single vs double quotes inside `onclick`/template literals) since nothing else will catch a mistake here.

- [ ] **Step 5: Commit**

```bash
git add static/backoffice.html src/api/tenants.py
git commit -m "feat(crm): backoffice tenant edit gets a CRM dropdown, base_url/auth_type move off it"
```

---

### Task 8: Backoffice — CRM management page

**Files:**
- Modify: `static/backoffice.html` (new page/section + navigation entry)

**Interfaces:**
- Consumes: `GET/POST/PATCH /api/v1/crms[/{id}]` (Task 5).

- [ ] **Step 1: Find the existing page-navigation pattern**

Run: `grep -n "function showPage\|function switchTab\|<nav\|class=\"tabs\"" static/backoffice.html | head -20` to find how this file already switches between sections (tenant list vs tenant detail, or similar), so the new CRM management section follows the same convention rather than inventing a new one.

- [ ] **Step 2: Add a CRM management section**

Add a new section (reachable via whatever nav mechanism Step 1 found) with:
- A table listing CRMs (`GET /crms`): id, name, base_url, tool count.
- A "New CRM" button opening a form: name, base_url, events_webhook_url_template, auth_type (dropdown: api_key/bearer), and one JSON `<textarea>` for the tools list (pre-filled with `[]` for a new CRM, or the current tools array — pretty-printed via `JSON.stringify(tools, null, 2)` — when editing an existing one).
- Save calls `POST /crms` (new) or `PATCH /crms/{id}` (edit), parsing the textarea with `JSON.parse` and showing a clear error message (not a silent failure) if the JSON is malformed.

Write the exact HTML/JS following this file's existing conventions (the `$()` helper, `escT`/`escAttr`, the `apiSend`/`apiTenant` fetch helpers, inline `<style>` classes already defined in the file) — mirror the structure of the existing tenant-list page as closely as possible since it already solves "list of entities + click to edit" in this same file.

- [ ] **Step 3: Manual verification**

Same as Task 7 Step 4 — `node --check` if available, plus a careful hand-trace of the new template literals/escaping.

- [ ] **Step 4: Commit**

```bash
git add static/backoffice.html
git commit -m "feat(crm): backoffice CRM management page (list/create/edit + JSON tool editor)"
```

---

### Task 9: Remove dead `PLATFORM_CRM_BASE_URL`/`PLATFORM_CRM_AUTH_TYPE`

**Files:**
- Modify: `src/config.py` (remove the `Secrets` fields)
- Modify: `.env.example` (remove the documentation lines)
- Test: none new — this is a deletion; existing tests must still pass.

- [ ] **Step 1: Grep-verify these are truly dead after Task 3's rewrite**

Run: `grep -rn "PLATFORM_CRM_BASE_URL\|PLATFORM_CRM_AUTH_TYPE" src/ tests/`
Expected: zero hits in `src/bootstrap.py` (Task 3 already removed the only reads); the only remaining hits should be the `Secrets` declarations in `src/config.py` and the `.env.example` comment lines, plus the migration file's `os.environ.get(...)` reads from Task 2 (those are fine — a migration reading a soon-to-be-removed env var ONE TIME during the upgrade that seeds the DB is expected and correct; it doesn't need the var to exist afterward).

- [ ] **Step 2: Remove from `src/config.py`**

Remove the `PLATFORM_CRM_BASE_URL: Optional[str] = None` and `PLATFORM_CRM_API_TOKEN`-adjacent... (note: `PLATFORM_CRM_API_TOKEN` was already removed earlier this session — only `PLATFORM_CRM_BASE_URL` and `PLATFORM_CRM_AUTH_TYPE` remain to remove now) lines from the `Secrets` class.

- [ ] **Step 3: Remove from `.env.example`**

Remove the corresponding `# PLATFORM_CRM_BASE_URL=...`/`# PLATFORM_CRM_AUTH_TYPE=...` comment lines.

- [ ] **Step 4: Re-verify**

Run: `grep -rn "PLATFORM_CRM_BASE_URL\|PLATFORM_CRM_AUTH_TYPE" src/config.py .env.example`
Expected: no output.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same known-baseline failures only (see Global Constraints), no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/config.py .env.example
git commit -m "chore(crm): remove dead PLATFORM_CRM_BASE_URL/AUTH_TYPE now the Crm entity owns these"
```

---

### Task 10: Docs

**Files:**
- Modify: `docs/chatbot.md` (CRM tools section)
- Modify: `docs/superpowers/specs/2026-07-23-crm-entity-design.md` (append an "Implemented" note — do not edit prior content)

- [ ] **Step 1: Update `docs/chatbot.md`**

Find the "CRM tools" section (added earlier this session) and update it to describe: tools now come from the tenant's linked `Crm` entity (`GET /api/v1/crms` for admin management), `source` on `/chat/tools/resolved` is `"crm_catalog"` (not `"platform_fallback"`), and per-tenant overrides (`chat_tools` rows, or the tenant's own `crm:api_token`/`crm:x_api_key`) still take precedence exactly as before.

- [ ] **Step 2: Append to the design spec**

Add a short "Implemented (2026-07-23)" note at the end of `docs/superpowers/specs/2026-07-23-crm-entity-design.md` pointing at this plan file and confirming the migration ran successfully against the real database (link the exact verification output from Task 2 Step 5).

- [ ] **Step 3: Commit**

```bash
git add docs/chatbot.md docs/superpowers/specs/2026-07-23-crm-entity-design.md
git commit -m "docs(crm): update chatbot.md + spec for the shipped CRM entity"
```

---

## Final verification (after all tasks)

Run the complete suite once more:
```bash
.venv/bin/python -m pytest tests/unit -q
```
Expected: only the known pre-existing baseline failures (see Global Constraints) — zero new failures across all 10 tasks combined.

Manually confirm against the real database (same one used in Task 2 Step 5): `GET /api/v1/chat/tools/resolved` for the `stage` tenant still returns `source: "crm_catalog"` with all 18 tools and correct `token_configured`/`x_api_key_configured` — i.e., the exact live CRM integration this whole feature grew out of debugging still works, unchanged in behavior, just now backed by the new entity instead of an env var + hardcoded dict.
