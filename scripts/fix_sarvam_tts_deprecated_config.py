"""Report/fix tenant rows pinned to the deprecated Sarvam ``bulbul:v2`` TTS
model (and its ``anushka`` speaker, which doesn't exist on ``bulbul:v3``).

Background: Sarvam deprecated ``bulbul:v2`` in 2026-09 ("Model 'bulbul:v2'
has been deprecated. Please use 'bulbul:v3' instead."), breaking outbound TTS
for every tenant. ``src/providers/tts/sarvam.py`` has been fixed to default
to ``bulbul:v3`` / a valid v3 speaker, which is enough for tenants whose
``pipeline_config.tts`` is null (they inherit the adapter default). It is NOT
enough for tenants that PIN the dead values explicitly in their DB row —
those need the row itself updated. This script finds and (optionally) fixes
those rows.

Scope: only ``model``/``voice_id`` are ever touched, and only inside the
``tts`` sub-object of ``pipeline_config`` — every other key (stt/llm/
telephony/vector_store/tts.language/tts.speed/tts.api_key_env/etc.) is
carried through unchanged. Non-Sarvam TTS tenants (a different ``provider``,
e.g. indicf5) are left alone entirely.

Speaker migration is deliberately narrow: the only automatic replacement this
script knows is ``anushka -> priya`` (the adapter's new ``DEFAULT_SPEAKER``),
because that is the ONE mapping actually verified — both by Sarvam's live API
(hand-tested, 2026-09) and by matching gender (anushka: female, priya:
female) so a campaign's gendered Hindi/Marathi grammar doesn't silently break.
Any other voice_id that turns out invalid under bulbul:v3 is reported under
"NEEDS MANUAL REVIEW" and never auto-fixed — this script does not guess a
replacement speaker (see the gender-uncertainty notes on
``src.providers.tts.sarvam._BULBUL_V3_SPEAKERS`` for why that would be
risky: a wrong gender produces grammatically wrong output on live calls).

Uses ``asyncpg`` (the driver used everywhere else in this repo, not
``psycopg2``). ``DATABASE_URL`` is resolved via ``src.config``'s settings
first (which read ``.env``), falling back to the raw process env var — a
plain ``python scripts/...`` run does not have ``.env`` exported into the
process environment. Mirrors ``scripts/reseed_crm_catalog_tools.py``.

Usage:
    python scripts/fix_sarvam_tts_deprecated_config.py
    python scripts/fix_sarvam_tts_deprecated_config.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # type: ignore[import-not-found]

from src.config import get_settings
from src.models.database import search_path_connect_args
from src.providers.tts.sarvam import _BULBUL_V3_SPEAKERS, DEFAULT_MODEL, DEFAULT_SPEAKER
from src.providers.vector_store.pgvector_store import _to_asyncpg_dsn

DEPRECATED_MODEL = "bulbul:v2"
_V3_SPEAKER_IDS = {s["voice_id"] for s in _BULBUL_V3_SPEAKERS}

# The only automatic speaker migration — see module docstring for why it's
# limited to this one, verified, gender-preserving mapping.
KNOWN_SPEAKER_REPLACEMENT: dict[str, str] = {
    "anushka": DEFAULT_SPEAKER,
}


@dataclass(frozen=True)
class TenantFix:
    tenant_id: str
    slug: str
    old_model: Optional[str]
    new_model: Optional[str]
    old_voice: Optional[str]
    new_voice: Optional[str]


@dataclass(frozen=True)
class TenantIssue:
    tenant_id: str
    slug: str
    voice_id: str


@dataclass(frozen=True)
class Plan:
    fixes: list[TenantFix] = field(default_factory=list)
    manual_review: list[TenantIssue] = field(default_factory=list)
    unaffected: list[str] = field(default_factory=list)  # human-readable notes


def compute_plan(tenants: list[dict[str, Any]]) -> Plan:
    """Pure planning over a snapshot of tenant rows — no DB access, safe to
    unit test.

    Each row: ``{"id": str, "slug": str, "pipeline_config": dict}`` — caller
    normalizes ``pipeline_config`` to a dict before calling (asyncpg returns
    json/jsonb columns as text on a bare connection).
    """
    fixes: list[TenantFix] = []
    manual: list[TenantIssue] = []
    unaffected: list[str] = []

    for row in tenants:
        pc = row.get("pipeline_config") or {}
        tts = pc.get("tts")
        if tts is not None and not isinstance(tts, dict):
            continue  # malformed tts value — not this script's problem to fix
        tts = tts or {}  # missing key or explicit JSON null: report as "null config" below

        provider = tts.get("provider")
        if provider not in (None, "sarvam"):
            continue  # a different TTS provider — out of scope for this fix

        old_model = tts.get("model")
        old_voice = tts.get("voice_id")

        new_model = old_model
        new_voice = old_voice
        changed = False

        if old_model == DEPRECATED_MODEL:
            new_model = DEFAULT_MODEL
            changed = True

        if old_voice and old_voice not in _V3_SPEAKER_IDS:
            replacement = KNOWN_SPEAKER_REPLACEMENT.get(old_voice.lower())
            if replacement:
                new_voice = replacement
                changed = True
            else:
                manual.append(TenantIssue(
                    tenant_id=row["id"], slug=row["slug"], voice_id=old_voice,
                ))

        if changed:
            fixes.append(TenantFix(
                tenant_id=row["id"], slug=row["slug"],
                old_model=old_model, new_model=new_model,
                old_voice=old_voice, new_voice=new_voice,
            ))
        elif not old_model and not old_voice:
            unaffected.append(
                f"{row['slug']}: tts.model/voice_id both null — inherits the "
                f"adapter default ({DEFAULT_MODEL}/{DEFAULT_SPEAKER}); no DB change needed"
            )
        else:
            unaffected.append(
                f"{row['slug']}: already v3-valid (model={old_model!r}, voice_id={old_voice!r})"
            )

    return Plan(fixes=fixes, manual_review=manual, unaffected=unaffected)


def _print_report(plan: Plan) -> None:
    print(f"FIX ({len(plan.fixes)})")
    for f in plan.fixes:
        print(f"  ~ {f.slug} (id={f.tenant_id})")
        if f.old_model != f.new_model:
            print(f"      model:    {f.old_model!r} -> {f.new_model!r}")
        if f.old_voice != f.new_voice:
            print(f"      voice_id: {f.old_voice!r} -> {f.new_voice!r}")

    print(f"\nNEEDS MANUAL REVIEW ({len(plan.manual_review)})")
    for i in plan.manual_review:
        print(f"  ? {i.slug} (id={i.tenant_id}): voice_id={i.voice_id!r} is not a "
              f"valid bulbul:v3 speaker and has no known safe replacement — fix by hand")

    print(f"\nNO CHANGE NEEDED ({len(plan.unaffected)})")
    for note in plan.unaffected:
        print(f"  = {note}")


def _resolve_database_url() -> str:
    """src.config's settings first (it reads .env), the raw process env var
    as a fallback — a plain ``python scripts/...`` invocation does not have
    ``.env`` exported into the process environment. Mirrors
    ``scripts/reseed_crm_catalog_tools.py``."""
    try:
        url = get_settings().secrets.DATABASE_URL
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default: dry run, nothing written)")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    db_url = _resolve_database_url()
    if not db_url:
        print(
            "ERROR: DATABASE_URL not set (checked src.config settings and the process env)",
            file=sys.stderr,
        )
        return 1

    dsn = _to_asyncpg_dsn(db_url)
    conn = await asyncpg.connect(dsn, **search_path_connect_args(db_url))
    try:
        rows = await conn.fetch("SELECT id, slug, pipeline_config FROM tenants ORDER BY slug")
        tenants = [
            {
                "id": row["id"],
                "slug": row["slug"],
                # asyncpg returns json/jsonb columns as text on a bare
                # connection (no codec registered) — normalize to a dict.
                "pipeline_config": json.loads(row["pipeline_config"])
                if isinstance(row["pipeline_config"], str)
                else (row["pipeline_config"] or {}),
            }
            for row in rows
        ]

        plan = compute_plan(tenants)
        _print_report(plan)

        if not args.apply:
            print("\nDry run — nothing written. Re-run with --apply to write.")
            return 0

        if not plan.fixes:
            print("\nNothing to apply — no fixable rows found. No write transaction opened.")
            return 0

        by_id = {t["id"]: t["pipeline_config"] for t in tenants}
        async with conn.transaction():
            for f in plan.fixes:
                pc = by_id[f.tenant_id]
                pc["tts"]["model"] = f.new_model
                pc["tts"]["voice_id"] = f.new_voice
                await conn.execute(
                    "UPDATE tenants SET pipeline_config = $1 WHERE id = $2",
                    json.dumps(pc), f.tenant_id,
                )

        print(f"\nDone — {len(plan.fixes)} tenant(s) updated.")
        print("  updated: " + ", ".join(f.slug for f in plan.fixes))
        if plan.manual_review:
            print(
                f"\n  {len(plan.manual_review)} tenant(s) still need manual review "
                "(see NEEDS MANUAL REVIEW above) — untouched by this run."
            )
        return 0
    finally:
        await conn.close()


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
