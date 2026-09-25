"""tests/unit/test_resend_session_closed.py

Tests for ``scripts/resend_session_closed.py``, the operator tool that
re-sends the ``session_closed`` BO webhook for chat sessions whose original
delivery failed. Uses an isolated aiosqlite in-memory engine (never the
process-global one in ``src.models.database``) plus a fake stand-in for
``send_bo_webhook`` so nothing here makes a real HTTP call or depends on
tenant-secret decryption (``VOX_SECRET_KEY``).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import scripts.resend_session_closed as script
from src.api import chat_webhooks
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.tenant import Tenant
from src.models.webhook_outbox import STATUS_DELIVERED, STATUS_PENDING, WebhookOutbox


@pytest_asyncio.fixture
async def sm(monkeypatch):
    """Isolated in-memory DB, wired into the script the same way the real
    process would be (get_engine/get_sessionmaker), but pointed at a private
    engine so this test never touches src.models.database's global one."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    monkeypatch.setattr(script, "_resolve_database_url", lambda: "sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(script, "get_engine", lambda url=None: None)
    monkeypatch.setattr(script, "get_sessionmaker", lambda: sessionmaker)

    async with sessionmaker() as db:
        db.add(Tenant(id="t1", slug="acme", name="Acme", pipeline_config={}))
        await db.commit()

    yield sessionmaker
    await engine.dispose()


class _FakeSender:
    """Records every call; ``result`` picks what it returns (per-call by
    session_id, or a single default for every call)."""

    def __init__(self, default: bool = True, per_session: dict[str, bool] | None = None):
        self.calls: list[tuple] = []
        self.default = default
        self.per_session = per_session or {}

    async def __call__(self, tenant, event_type, payload) -> bool:
        self.calls.append((tenant, event_type, payload))
        return self.per_session.get(payload.get("session_id"), self.default)


async def _add_session(
    sm, session_id: str, *, status: str = "ended", mode: str = "closed",
    claimed_by: str | None = None, summary: str | None = "customer resolved",
    tenant_id: str = "t1",
) -> None:
    async with sm() as db:
        db.add(ChatSession(
            id=session_id, tenant_id=tenant_id, language="hi", status=status,
            mode=mode, extra_data={}, message_count=2, claimed_by=claimed_by,
            summary=summary, ended_at=datetime.now(timezone.utc).replace(tzinfo=None) if status == "ended" else None,
        ))
        db.add(ChatMessage(session_id=session_id, role="customer", type="text", content="hi"))
        db.add(ChatMessage(session_id=session_id, role="agent", type="text", content="hello"))
        await db.commit()


async def _add_outbox_row(
    sm, *, row_id: str, session_id: str, tenant_id: str = "t1",
    status: str = STATUS_PENDING, attempts: int = 1, last_error: str | None = "http_503",
) -> None:
    async with sm() as db:
        db.add(WebhookOutbox(
            id=row_id, tenant_id=tenant_id, session_id=session_id, event_type="session_closed",
            url="https://crm.example/hook", body={"event": "session_closed", "session_id": session_id},
            attempts=attempts, next_attempt_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status=status, last_error=last_error,
        ))
        await db.commit()


async def test_prints_redacted_db_target(sm, monkeypatch, capsys):
    """M3: the DB host/name/schema (never the raw URL, never credentials) is
    printed at startup in both dry-run and --apply."""
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    monkeypatch.setattr(script, "_resolve_database_url",
                         lambda: "postgresql+asyncpg://user:s3cr3t@db.internal:5432/voicebot")
    await _add_session(sm, "cs_1")

    rc = await script._run(script.parse_args(["cs_1"]))

    assert rc == 0
    out = capsys.readouterr().out
    assert "database:" in out
    assert "db.internal:5432/voicebot" in out
    assert "s3cr3t" not in out
    assert "user:" not in out


async def test_pending_outbox_row_is_skipped_without_force(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")
    await _add_outbox_row(sm, row_id="wh_1", session_id="cs_1")

    rc = await script._run(script.parse_args(["cs_1", "--apply"]))

    assert rc == 0
    assert fake.calls == []  # never sent -- the pending outbox row blocked it
    out = capsys.readouterr().out
    assert "webhook_outbox: status=pending attempts=1" in out
    assert "SKIPPED" in out
    assert "pass --force" in out


async def test_pending_outbox_row_is_sent_with_force(sm, monkeypatch):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")
    await _add_outbox_row(sm, row_id="wh_1", session_id="cs_1")

    rc = await script._run(script.parse_args(["cs_1", "--apply", "--force"]))

    assert rc == 0
    assert len(fake.calls) == 1  # --force overrides the pending-row block


async def test_delivered_outbox_row_does_not_block_resend(sm, monkeypatch):
    """A terminal (delivered/dead) row is purely informational -- only a
    still-pending row blocks a resend."""
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")
    await _add_outbox_row(sm, row_id="wh_1", session_id="cs_1", status=STATUS_DELIVERED, last_error=None)

    rc = await script._run(script.parse_args(["cs_1", "--apply"]))

    assert rc == 0
    assert len(fake.calls) == 1


async def test_dry_run_shows_no_outbox_row_when_none_exists(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")

    rc = await script._run(script.parse_args(["cs_1"]))

    assert rc == 0
    out = capsys.readouterr().out
    assert "webhook_outbox: none" in out


async def test_dry_run_sends_nothing(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")

    rc = await script._run(script.parse_args(["cs_1"]))

    assert rc == 0
    assert fake.calls == []
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "would send" in out


async def test_apply_sends_once_per_ended_session(sm, monkeypatch):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")
    await _add_session(sm, "cs_2", claimed_by="agent-42")

    rc = await script._run(script.parse_args(["cs_1", "cs_2", "--apply"]))

    assert rc == 0
    assert len(fake.calls) == 2
    by_session = {c[2]["session_id"]: c for c in fake.calls}

    tenant1, event1, payload1 = by_session["cs_1"]
    assert event1 == "session_closed"
    assert payload1["mode_at_close"] == "ai"  # never claimed -> bot-only close
    assert payload1["summary"] == "customer resolved"
    assert [m["role"] for m in payload1["transcript"]] == ["customer", "agent"]
    assert tenant1.slug == "acme"

    _, _, payload2 = by_session["cs_2"]
    assert payload2["mode_at_close"] == "human"  # claimed_by set -> human close


async def test_apply_uses_empty_string_when_summary_is_none(sm, monkeypatch):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1", summary=None)

    await script._run(script.parse_args(["cs_1", "--apply"]))

    assert fake.calls[0][2]["summary"] == ""


async def test_non_ended_session_is_skipped_not_sent(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_live", status="active", mode="ai")

    rc = await script._run(script.parse_args(["cs_live", "--apply"]))

    assert rc == 0
    assert fake.calls == []
    out = capsys.readouterr().out
    assert "SKIPPED" in out
    assert "not 'ended'" in out


async def test_unknown_session_id_is_reported_not_a_crash(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)

    rc = await script._run(script.parse_args(["cs_does_not_exist", "--apply"]))

    assert rc == 0
    assert fake.calls == []
    out = capsys.readouterr().out
    assert "NOT FOUND" in out


async def test_apply_reports_failure_and_nonzero_exit(sm, monkeypatch):
    fake = _FakeSender(default=False)
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")

    rc = await script._run(script.parse_args(["cs_1", "--apply"]))

    assert rc == 1
    assert len(fake.calls) == 1


async def test_mixed_batch_dry_run_counts(sm, monkeypatch, capsys):
    fake = _FakeSender()
    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", fake)
    await _add_session(sm, "cs_1")
    await _add_session(sm, "cs_live", status="active", mode="ai")

    rc = await script._run(script.parse_args(["cs_1", "cs_live", "cs_missing"]))

    assert rc == 0
    assert fake.calls == []


def test_collect_ids_dedupes_across_args_and_file(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("cs_2\n# comment\n\ncs_3\ncs_1\n")
    args = script.parse_args(["cs_1", "cs_2", "--file", str(f)])
    assert script._collect_ids(args) == ["cs_1", "cs_2", "cs_3"]
