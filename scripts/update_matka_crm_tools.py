"""One-off patch: fix/add Matka tools on the betstudio CRM tool catalog.

- get_matka_bids: endpoint /players/{user_id}/matka-bids -> /players/{user_id}/matka/bids
- get_matka_config: renamed to get_matka_markets, endpoint
  /operators/{operator_id}/matka-config -> /operators/{operator_id}/matka/markets
- get_operator_games_config: description's "use get_matka_config instead" reference
  updated to "use get_matka_markets instead"
- get_matka_result: new tool, /operators/{operator_id}/matka-results

Run against whichever DB DATABASE_URL points at (staging first, then prod once verified):

    DATABASE_URL=<staging-db-url> python scripts/update_matka_crm_tools.py
    DATABASE_URL=<staging-db-url> python scripts/update_matka_crm_tools.py --dry-run

Defaults to crm_id=betstudio; override with --crm-id if needed.
"""

import argparse
import json
import os
import sys

GET_MATKA_RESULT_DESCRIPTION = (
    "Get the declared/settled result for a specific Matka market and "
    "session (e.g. today's Rajdhani Day, Kalyan, Milan Day number) — "
    "independent of whether the customer has a bet placed on it, and "
    "usable even for a customer who isn't logged in with an active bet. "
    "Call this whenever the customer asks for a market's result directly "
    "(e.g. 'Rajdhani Day ka result kya aya hai', 'Kalyan ka aaj ka number "
    "kya hai', 'is market ka result declare hua kya', 'Milan Day open "
    "result kya tha'). "
    "NOTE: this does NOT know which bets a player placed or whether they "
    "won/lost — for the customer's OWN bid outcome, settlement status, or "
    "payout, call get_matka_bids instead."
)
GET_MATKA_RESULT_PARAMETERS = {
    "operator_id": {"type": "string", "source": "session", "description": "Operator identifier"},
    "market": {"type": "string", "source": "llm",
               "description": "Market name (e.g. 'Kalyan', 'Milan Day', 'Rajdhani Day', 'Starline')"},
    "date": {"type": "string", "source": "llm",
             "description": "Optional: date to fetch the result for (YYYY-MM-DD). Omit for today's/latest declared result."},
}

GAMES_CONFIG_OLD_REF = "For Matka-specific details (markets, odds, timing, bet types) use get_matka_config instead."
GAMES_CONFIG_NEW_REF = "For Matka-specific details (markets, odds, timing, bet types) use get_matka_markets instead."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crm-id", default="betstudio")
    parser.add_argument("--dry-run", action="store_true", help="Print intended changes without writing")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    import psycopg2
    import psycopg2.extras

    dsn = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )

    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute(
            "SELECT id, name, endpoint, description FROM crm_tools WHERE crm_id = %s",
            (args.crm_id,),
        )
        rows = {r["name"]: r for r in cur.fetchall()}

        actions = []

        bids = rows.get("get_matka_bids")
        if bids:
            actions.append((
                "UPDATE get_matka_bids endpoint",
                "UPDATE crm_tools SET endpoint = %s WHERE id = %s",
                ("/players/{user_id}/matka/bids", bids["id"]),
            ))
        else:
            print("WARNING: get_matka_bids not found for this crm_id — skipping", file=sys.stderr)

        config = rows.get("get_matka_config")
        if config:
            actions.append((
                "RENAME get_matka_config -> get_matka_markets + endpoint",
                "UPDATE crm_tools SET name = %s, endpoint = %s WHERE id = %s",
                ("get_matka_markets", "/operators/{operator_id}/matka/markets", config["id"]),
            ))
        else:
            print("WARNING: get_matka_config not found for this crm_id — skipping rename", file=sys.stderr)

        games_config = rows.get("get_operator_games_config")
        if games_config and GAMES_CONFIG_OLD_REF in games_config["description"]:
            new_desc = games_config["description"].replace(GAMES_CONFIG_OLD_REF, GAMES_CONFIG_NEW_REF)
            actions.append((
                "FIX get_operator_games_config cross-reference",
                "UPDATE crm_tools SET description = %s WHERE id = %s",
                (new_desc, games_config["id"]),
            ))
        elif games_config:
            print("NOTE: get_operator_games_config description doesn't contain the expected "
                  "old reference text — leaving it untouched", file=sys.stderr)

        if "get_matka_result" in rows:
            print("NOTE: get_matka_result already exists for this crm_id — skipping insert", file=sys.stderr)
        else:
            actions.append((
                "INSERT get_matka_result",
                "INSERT INTO crm_tools (crm_id, name, description, endpoint, method, parameters) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (args.crm_id, "get_matka_result", GET_MATKA_RESULT_DESCRIPTION,
                 "/operators/{operator_id}/matka-results", "GET", json.dumps(GET_MATKA_RESULT_PARAMETERS)),
            ))

        if not actions:
            print("Nothing to do.")
            return

        print(f"crm_id={args.crm_id}: {len(actions)} change(s) to apply" + (" (dry run)" if args.dry_run else ""))
        for label, sql, params in actions:
            print(f"  - {label}")
            if not args.dry_run:
                cur.execute(sql, params)

        if args.dry_run:
            print("\nDry run — no changes written.")
        else:
            conn.commit()
            print(f"\nDone — {len(actions)} change(s) applied.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
