"""Probe how Stringee authenticates a recording download — for a real call id.

Given a Stringee ``callId`` and the tenant's Stringee API Key SID/Secret, this
fetches ``https://api.stringee.com/v1/call/recording/<callId>`` several ways and
prints the HTTP status + size for each, so we can see definitively which auth
Stringee accepts (header vs ``access_token`` query) — i.e. validate the
``_download_stringee_recording`` fix (PR #100).

Credentials (first match wins):
  1. --sid / --secret CLI args
  2. --tenant <slug>  → decrypt the tenant's Stringee creds from the DB
                        (needs DATABASE_URL + VOX_SECRET_KEY in .env)
  3. env STRINGEE_API_KEY_SID / STRINGEE_API_KEY_SECRET

Usage:
    python scripts/stringee_recording_probe.py --call-id call-vn-1-... --tenant stage
    python scripts/stringee_recording_probe.py --call-id call-vn-1-... --sid SK... --secret ...

A real GET hits the Stringee API; recordings are only available for a window
after the call, so use a recent call id.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECORDING_BASE = "https://api.stringee.com/v1/call/recording"


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    vals: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = (part.strip() for part in line.split("=", 1))
        val = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1)) or os.environ.get(m.group(1), ""), val)
        vals[key] = val
    for k, v in vals.items():
        os.environ.setdefault(k, v)


async def _creds_from_tenant(slug: str) -> tuple[str, str]:
    """Decrypt the tenant's active Stringee account_sid/auth_token from the DB."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from src.auth import secrets as crypto
    from src.config_tenant import TenantTelephonyConfig
    from src.models.tenant import Tenant, TenantSecret

    schema = os.environ.get("VOX_DB_SCHEMA", "voicebot")
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    p = urlsplit(url)
    url = urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    eng = create_async_engine(url, connect_args={
        "ssl": ssl.create_default_context(), "server_settings": {"search_path": schema}})
    sm = async_sessionmaker(eng, expire_on_commit=False)
    async with sm() as s:
        t = (await s.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one()
        tel = TenantTelephonyConfig(**((t.pipeline_config or {}).get("telephony") or {}))
        creds = tel.active_creds()
        rows = {r.name: r.value_encrypted for r in
                (await s.execute(select(TenantSecret).where(TenantSecret.tenant_id == t.id))).scalars()}
        sid = crypto.decrypt(rows[creds.account_sid_env])
        secret = crypto.decrypt(rows[creds.auth_token_env])
    await eng.dispose()
    return sid, secret


async def _probe(label: str, url: str, headers: dict) -> None:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
        try:
            r = await c.get(url, headers=headers)
            body = "" if r.is_success else f" body={r.text[:200]!r}"
            print(f"  [{label}] HTTP {r.status_code}  bytes={len(r.content)}"
                  f"  final={r.url.scheme}://{r.url.host}{r.url.path}{body}")
        except httpx.HTTPError as e:
            print(f"  [{label}] ERROR {e}")


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe Stringee recording-download auth")
    ap.add_argument("--call-id", required=True)
    ap.add_argument("--tenant", default=None, help="resolve creds from this tenant's DB row")
    ap.add_argument("--sid", default=None)
    ap.add_argument("--secret", default=None)
    args = ap.parse_args(argv)
    _load_dotenv()

    from src.providers.telephony.stringee import mint_server_token

    if args.sid and args.secret:
        sid, secret = args.sid, args.secret
    elif args.tenant:
        sid, secret = await _creds_from_tenant(args.tenant)
    else:
        sid = os.environ.get("STRINGEE_API_KEY_SID")
        secret = os.environ.get("STRINGEE_API_KEY_SECRET")
    if not (sid and secret):
        print("error: no Stringee creds (pass --sid/--secret, --tenant, or set env)", file=sys.stderr)
        return 2

    token = mint_server_token(sid, secret)
    print(f"call_id={args.call_id}  keySid={sid[:6]}…  token={token[:10]}…\n")

    https = f"{RECORDING_BASE}/{args.call_id}"
    http = https.replace("https://", "http://", 1)
    hdr = {"X-STRINGEE-AUTH": token}
    q = f"{https}?access_token={token}"

    print("Probing recording download (each follows redirects):")
    await _probe("https + header only", https, hdr)
    await _probe("https + access_token query only", q, {})
    await _probe("https + header + query (what the app sends)", q, hdr)
    await _probe("http + header + query (shows 301/query handling)",
                 f"{http}?access_token={token}", hdr)
    print("\nThe variant that returns HTTP 200 with bytes>0 is the one Stringee wants.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
