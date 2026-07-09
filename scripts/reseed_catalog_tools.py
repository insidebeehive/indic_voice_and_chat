"""Update description and parameters for catalog tools across all tenants.

Patches only the fields sourced from the catalog (description, parameters).
Leaves endpoint, method, auth_type, and auth_config untouched so existing
CRM integrations keep working.

Usage:
    python scripts/reseed_catalog_tools.py
    python scripts/reseed_catalog_tools.py --tools get_game get_operator_games_config
    python scripts/reseed_catalog_tools.py --tenant mgmstage
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot.catalog import ALL_TOOLS


def main() -> None:
    parser = argparse.ArgumentParser(description="Reseed catalog tool descriptions/parameters in DB")
    parser.add_argument("--tools", nargs="+", default=list(ALL_TOOLS.keys()),
                        help="Tool names to update (default: all catalog tools)")
    parser.add_argument("--tenant", default=None,
                        help="Limit to a specific tenant slug (default: all tenants)")
    args = parser.parse_args()

    unknown = [t for t in args.tools if t not in ALL_TOOLS]
    if unknown:
        print(f"ERROR: unknown tool names: {unknown}", file=sys.stderr)
        print(f"Available: {list(ALL_TOOLS.keys())}", file=sys.stderr)
        sys.exit(1)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    import psycopg2
    import psycopg2.extras

    # Convert SQLAlchemy URL to psycopg2 DSN
    dsn = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )

    print(f"Reseeding: {args.tools}" + (f" for tenant={args.tenant}" if args.tenant else " for all tenants"))

    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        placeholders = ",".join(["%s"] * len(args.tools))
        if args.tenant:
            cur.execute(
                f"""SELECT ct.id, ct.name, t.slug
                    FROM chat_tools ct
                    JOIN tenants t ON t.id = ct.tenant_id
                    WHERE ct.name IN ({placeholders}) AND t.slug = %s""",
                args.tools + [args.tenant]
            )
        else:
            cur.execute(
                f"""SELECT ct.id, ct.name, t.slug
                    FROM chat_tools ct
                    JOIN tenants t ON t.id = ct.tenant_id
                    WHERE ct.name IN ({placeholders})""",
                args.tools
            )

        rows = cur.fetchall()
        if not rows:
            print("No matching tools found in DB.")
            return

        updated = 0
        for row in rows:
            spec = ALL_TOOLS[row["name"]]
            cur.execute(
                "UPDATE chat_tools SET description = %s, parameters = %s WHERE id = %s",
                (spec["description"], json.dumps(spec["parameters"]), row["id"])
            )
            print(f"  updated {row['name']} (tenant={row['slug']})")
            updated += 1

        conn.commit()
        print(f"\nDone — {updated} tool(s) updated.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
