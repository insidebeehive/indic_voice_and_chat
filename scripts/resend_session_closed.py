"""Re-send the ``session_closed`` BO webhook for chat sessions whose original
delivery failed (e.g. the CRM endpoint timed out) even though our own
``chat_sessions`` row is already ``ended`` — so the CRM ticket never got the
signal to close and stays open forever.

This is a one-off operator tool, not something the app calls itself. Point it
at one or more ``cs_...`` session ids and it will, by default, only report
what it *would* send (dry run). Pass ``--apply`` to actually send.

**Refuses to touch a live session.** Only a session whose ``status`` is
already ``ended`` is eligible — a session that is still ``active``/
``escalated`` is skipped with a reason, never closed by this script. Closing
a session is exclusively the job of the live chat path (``src/api/chat.py``).

**Payload.** The live path's own resend helper is
``src.api.chat._send_close_webhook``, which rebuilds the transcript from
``chat_messages`` and calls ``src.api.chat_webhooks.send_bo_webhook``. That
helper is deliberately best-effort: it swallows every exception and does not
propagate ``send_bo_webhook``'s success/failure bool, so a caller that needs
to know whether the resend actually landed (this script does — it has to
report per-session OK/failed) can't get that out of it. Rather than import it
just to throw away the one thing this script needs, this script reimplements
its payload construction — {"session_id", "mode_at_close", "summary",
"transcript"}, transcript = every ``chat_messages`` row for the session
ordered by id, each as {"role", "text", "ts"} — line for line the same as
``_send_close_webhook`` (src/api/chat.py, search for that name), and calls
``send_bo_webhook`` directly to get its bool back. This also avoids importing
``src.api.chat`` at all, which matters right now because that module has
in-flight edits elsewhere. If ``_send_close_webhook``'s payload shape ever
changes, update the ``_build_payload`` function below to match.

``mode_at_close`` is not stored anywhere after the fact — ``_end_session``
always overwrites ``chat_sessions.mode`` to ``"closed"`` when a session ends,
so the row can no longer tell you whether it was AI-only or had a human
agent. The one durable signal is ``claimed_by``: it's set exactly once, by
``POST /chat/sessions/{id}/claim`` (src/api/chat.py), and never cleared. So
this script uses ``mode_at_close = "human" if row.claimed_by else "ai"``,
matching how the live paths pick it (the human-mode WS close path hardcodes
``"human"``; every bot-only close path hardcodes ``"ai"``).

**Idempotency.** Whether the CRM treats a repeated ``session_closed`` for a
session_id it already closed as a no-op is a CRM-side contract this script
cannot verify — it isn't visible from here. Confirm with the CRM team before
relying on that, especially before a bulk ``--apply`` run.

**webhook_outbox awareness.** A session_closed delivery that fails in-line
now (see ``src/models/webhook_outbox.py``) gets queued for durable retry by
the live app itself — a background loop keeps retrying it for up to ~24h.
This script shows each session's current ``webhook_outbox`` row (if any) in
its plan, and — since sending here as well would race that background retry
and could double-deliver — refuses to send (in ``--apply``) for a session
with a still-``pending`` row unless ``--force`` is also passed. A
``delivered``/``dead`` row (or no row at all) never blocks a resend.

Required env vars:
    DATABASE_URL      Postgres URL. Resolved the same way as
                       scripts/reseed_crm_catalog_tools.py: src.config's
                       settings first (which reads .env), the raw process env
                       var as fallback — an explicit ``DATABASE_URL=...`` on
                       the command line still wins over whatever settings
                       would otherwise resolve to.
    VOX_SECRET_KEY     Fernet key used to decrypt this tenant's rows in
                       tenant_secrets (src/auth/secrets.py). Only per-tenant
                       TELEPHONY secrets live there — the webhook signing
                       secret is normally the platform-wide
                       EVENTS_WEBHOOK_SECRET env var instead — but
                       tenant_context_from_row decrypts every declared secret
                       for the tenant regardless of kind, so leaving this
                       unset means those rows fail to decrypt (logged, not
                       fatal) rather than actually being needed for the
                       webhook call itself.
    EVENTS_WEBHOOK_SECRET (optional)
                       Platform-wide fallback HMAC signing secret, used when
                       the tenant has no per-tenant
                       ``events_webhook_secret_env`` configured. Same var the
                       live app uses.

Usage:
    # Dry run (default) — prints what would be sent, sends nothing.
    python scripts/resend_session_closed.py cs_abc123 cs_def456

    # From a file, one session id per line (blank lines/`#` comments ignored).
    python scripts/resend_session_closed.py --file ids.txt

    # Actually send.
    python scripts/resend_session_closed.py cs_abc123 --apply

    # Mix both sources; duplicates are de-duplicated, order preserved.
    python scripts/resend_session_closed.py cs_abc123 --file ids.txt --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.auth.db_resolver import tenant_context_from_row
from src.models.chat import ChatMessage, ChatSession
from src.models.database import get_engine, get_schema, get_sessionmaker
from src.models.tenant import Tenant
from src.models.webhook_outbox import STATUS_PENDING, WebhookOutbox


@dataclass(frozen=True)
class OutboxStatus:
    """The most recent ``webhook_outbox`` row for a session's session_closed
    event, if any — see M3 in the coordinator's review: this script must show
    it and refuse to resend over a still-``pending`` one without ``--force``."""

    status: str
    attempts: int
    last_error: Optional[str]


@dataclass(frozen=True)
class SessionPlan:
    """Everything needed to describe/send session_closed for one session."""

    session_id: str
    tenant_id: str
    tenant_slug: str
    status: str
    stored_mode: str
    mode_at_close: str
    ended_at: Optional[datetime]
    message_count: int
    has_summary: bool
    ticket_id: Optional[str]
    outbox: Optional[OutboxStatus]


def _resolve_database_url() -> str:
    """src.config's settings first (it reads .env), the raw process env var
    as a fallback — mirrors ``_resolve_database_url`` in
    scripts/reseed_crm_catalog_tools.py and
    tests/unit/test_crm_catalog_seeding.py. A plain ``python scripts/...``
    invocation does not have ``.env`` exported into the process environment,
    so resolving through settings first means the caller doesn't have to
    remember to export DATABASE_URL by hand. An explicit
    ``DATABASE_URL=<url> python scripts/...`` still wins, since settings
    itself reads the env var when building the URL."""
    try:
        from src.config import get_settings

        url = get_settings().database.url
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


def _describe_db_target(url: str) -> str:
    """Host/port/dbname + schema only, never credentials — printed once at
    startup (both dry-run and ``--apply``, M3) so an operator can confirm
    which database this run will touch before anything is sent."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
        dbname = parts.path.lstrip("/") or "?"
    except Exception:
        host, port, dbname = "?", "", "?"
    try:
        schema = get_schema(url) or "(default)"
    except Exception:
        schema = "?"
    return f"{host}{port}/{dbname} schema={schema}"


def _outbox_block_reason(plan: SessionPlan, *, force: bool) -> Optional[str]:
    """Why this session's resend should be refused because of its
    webhook_outbox state — None if it's fine to send. A still-pending row
    means the live app's own background loop is already retrying this exact
    delivery; sending here too would race it and risk a double-delivery to
    the CRM. delivered/dead rows (or no row at all) never block a resend."""
    if plan.outbox is not None and plan.outbox.status == STATUS_PENDING and not force:
        return (
            f"a pending webhook_outbox row already queues this session for durable "
            f"retry (attempts={plan.outbox.attempts}) — pass --force to resend anyway"
        )
    return None


def _ticket_id(row: ChatSession) -> Optional[str]:
    """Mirrors ``_ticket_id_from_row`` in src/api/chat.py (not imported, to
    avoid pulling in that module) — the CRM ticket id stashed on
    ``extra_data['crm_ticket_id']`` at session-creation time. Often absent."""
    return (row.extra_data or {}).get("crm_ticket_id")


def _build_payload(transcript: list[dict], plan: SessionPlan, summary: str) -> dict:
    """Identical shape to ``src.api.chat._send_close_webhook``'s payload —
    see the module docstring for why this isn't a call to that function."""
    return {
        "session_id": plan.session_id,
        "mode_at_close": plan.mode_at_close,
        "summary": summary,
        "transcript": transcript,
    }


async def _load_transcript(sm, session_id: str) -> list[dict]:
    async with sm() as db:
        msgs = (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id)
            )
        ).scalars().all()
    return [
        {"role": m.role, "text": m.content, "ts": m.created_at.isoformat() if m.created_at else None}
        for m in msgs
    ]


async def _plan_for_session(sm, session_id: str) -> tuple[Optional[SessionPlan], Optional[str]]:
    """Returns (plan, skip_reason). Exactly one is non-None. ``plan`` is None
    for an unknown id or a session that isn't ``ended`` (never touched)."""
    async with sm() as db:
        row = await db.get(ChatSession, session_id)
        if row is None:
            return None, "NOT FOUND — no chat_sessions row with this id"
        if row.status != "ended":
            return None, f"SKIPPED — status={row.status!r}, not 'ended' (refusing to touch a live session)"

        tenant_row = (
            await db.execute(
                select(Tenant)
                .where(Tenant.id == row.tenant_id)
                .options(selectinload(Tenant.phone_numbers), selectinload(Tenant.secrets))
            )
        ).scalar_one_or_none()
        if tenant_row is None:
            return None, f"SKIPPED — tenant_id={row.tenant_id!r} has no tenants row"

        outbox_row = (
            await db.execute(
                select(WebhookOutbox)
                .where(WebhookOutbox.session_id == session_id, WebhookOutbox.event_type == "session_closed")
                .order_by(WebhookOutbox.created_at.desc())
                .limit(1)
            )
        ).scalars().first()
        outbox = (
            OutboxStatus(status=outbox_row.status, attempts=outbox_row.attempts, last_error=outbox_row.last_error)
            if outbox_row is not None else None
        )

        plan = SessionPlan(
            session_id=row.id,
            tenant_id=tenant_row.id,
            tenant_slug=tenant_row.slug,
            status=row.status,
            stored_mode=row.mode,
            # See module docstring: row.mode is always "closed" by the time a
            # session is ended, so it can't tell us ai vs human -- claimed_by
            # is the one field that still can.
            mode_at_close="human" if row.claimed_by else "ai",
            ended_at=row.ended_at,
            message_count=row.message_count or 0,
            has_summary=bool(row.summary),
            ticket_id=_ticket_id(row),
            outbox=outbox,
        )
        return plan, None


def _print_plan(plan: SessionPlan, *, force: bool) -> None:
    print(f"session {plan.session_id}")
    print(f"  tenant: slug={plan.tenant_slug!r} id={plan.tenant_id!r}")
    print(f"  status={plan.status} mode(stored)={plan.stored_mode} mode_at_close(to send)={plan.mode_at_close}")
    print(f"  ended_at={plan.ended_at} messages={plan.message_count} has_summary={plan.has_summary}")
    if plan.ticket_id:
        print(f"  crm_ticket_id={plan.ticket_id}")
    if plan.outbox is None:
        print("  webhook_outbox: none")
    else:
        err = f" last_error={plan.outbox.last_error!r}" if plan.outbox.last_error else ""
        print(f"  webhook_outbox: status={plan.outbox.status} attempts={plan.outbox.attempts}{err}")
    block_reason = _outbox_block_reason(plan, force=force)
    if block_reason:
        print(f"  would SKIP: {block_reason}")
    else:
        print(
            f"  would send: event=session_closed session_id={plan.session_id} "
            f"mode_at_close={plan.mode_at_close} summary={'<stored>' if plan.has_summary else '<empty>'}"
        )


def _read_ids_from_file(path: str) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("session_ids", nargs="*", help="chat_sessions.id values (cs_...)")
    parser.add_argument("--file", help="Path to a file with one session id per line")
    parser.add_argument("--apply", action="store_true",
                        help="Actually send the webhook (default: dry run, sends nothing)")
    parser.add_argument(
        "--force", action="store_true",
        help="Also resend sessions with a still-pending webhook_outbox row "
             "(by default those are skipped — the live app's own background "
             "loop is already retrying them; see the module docstring)",
    )
    return parser.parse_args(argv)


def _collect_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.session_ids)
    if args.file:
        ids.extend(_read_ids_from_file(args.file))
    # De-duplicate, preserve first-seen order.
    return list(dict.fromkeys(ids))


async def _run(args: argparse.Namespace) -> int:
    from src.api.chat_webhooks import send_bo_webhook

    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set (checked src.config settings and the process env)", file=sys.stderr)
        return 1

    ids = _collect_ids(args)
    if not ids:
        print("ERROR: no session ids given (pass them as args and/or --file)", file=sys.stderr)
        return 1

    # M3: printed unconditionally, dry-run and --apply alike, so an operator
    # sees which database this run touches before anything is sent — never
    # the raw URL (which can carry credentials), just host/port/dbname/schema.
    print(f"database: {_describe_db_target(db_url)}")
    print()

    get_engine(db_url)
    sm = get_sessionmaker()

    sent = 0
    failed = 0
    skipped = 0
    would_send = 0

    for session_id in ids:
        plan, skip_reason = await _plan_for_session(sm, session_id)
        if plan is None:
            print(f"session {session_id}: {skip_reason}")
            skipped += 1
            continue

        _print_plan(plan, force=args.force)
        block_reason = _outbox_block_reason(plan, force=args.force)

        if not args.apply:
            if block_reason is None:
                would_send += 1
            else:
                skipped += 1
            continue

        if block_reason is not None:
            print(f"  -> SKIPPED: {block_reason}")
            skipped += 1
            continue

        async with sm() as db:
            row = await db.get(ChatSession, session_id)
            tenant_row = (
                await db.execute(
                    select(Tenant)
                    .where(Tenant.id == plan.tenant_id)
                    .options(selectinload(Tenant.phone_numbers), selectinload(Tenant.secrets))
                )
            ).scalar_one()
            tenant_ctx = tenant_context_from_row(tenant_row)
            summary = row.summary or ""

        transcript = await _load_transcript(sm, session_id)
        payload = _build_payload(transcript, plan, summary)
        ok = await send_bo_webhook(tenant_ctx, "session_closed", payload)
        if ok:
            print("  -> OK")
            sent += 1
        else:
            print("  -> FAILED")
            failed += 1

    print()
    if args.apply:
        print(f"Done — {sent} sent, {failed} failed, {skipped} skipped.")
    else:
        print(f"Dry run — nothing sent. {would_send} would be sent, {skipped} skipped. Re-run with --apply to send.")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
