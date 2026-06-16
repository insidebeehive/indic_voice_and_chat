"""Diagnostic: verify the Stringee softphone key/secret DB rows.

Loads every tenant from the DB exactly as the resolver does, finds the ones on
the Stringee telephony provider, and reports what `mint_browser_credentials`
would actually use for the JWT `iss` (the API Key SID Stringee rejected with
API_KEY_SID_NOT_FOUND). Masks values; flags missing rows + whitespace.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.auth.db_resolver import tenant_context_from_row
from src.models.database import get_sessionmaker  # noqa: E402  (after load_dotenv)
from src.models.tenant import Tenant


def mask(v):
    if v is None:
        return "<MISSING>"
    if v == "":
        return "<EMPTY STRING>"
    show = v if len(v) <= 8 else f"{v[:4]}…{v[-3:]}"
    flags = []
    if v != v.strip():
        flags.append("HAS LEADING/TRAILING WHITESPACE")
    if "\n" in v or "\r" in v:
        flags.append("HAS NEWLINE")
    suffix = f"  ⚠ {', '.join(flags)}" if flags else ""
    return f"{show!r} (len={len(v)}){suffix}"


async def main():
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (await session.execute(
            select(Tenant).options(
                selectinload(Tenant.phone_numbers),
                selectinload(Tenant.api_keys),
                selectinload(Tenant.secrets),
            )
        )).scalars().all()

    print(f"loaded {len(rows)} tenant(s)\n")
    for t in rows:
        ctx = tenant_context_from_row(t)
        tel = ctx.settings.pipeline.telephony
        provider = (tel.provider or "").lower()
        print(f"── tenant {t.slug!r} (id={t.id}) provider={provider!r}")
        if provider != "stringee":
            print("   (not stringee — skipping)\n")
            continue

        creds = tel.active_creds()
        sid_name = creds.account_sid_env
        sec_name = creds.auth_token_env
        print(f"   config account_sid_env = {sid_name!r}")
        print(f"   config auth_token_env  = {sec_name!r}")

        stored_names = sorted(s.name for s in t.secrets)
        print(f"   tenant_secrets rows    = {stored_names}")

        sid = ctx.secret(sid_name) or os.environ.get("STRINGEE_API_KEY_SID")
        sec = ctx.secret(sec_name) or os.environ.get("STRINGEE_API_KEY_SECRET")
        print(f"   → JWT iss (API Key SID) = {mask(sid)}")
        print(f"   → signing secret        = {mask(sec)}")

        if sid_name not in stored_names:
            print(f"   ⚠ no row named {sid_name!r} — minting falls back to env / None")
        if sec_name not in stored_names:
            print(f"   ⚠ no row named {sec_name!r} — minting falls back to env / None")
        print()


if __name__ == "__main__":
    asyncio.run(main())
