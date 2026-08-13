"""List every schema in the target DB that actually has a chat_sessions
table, and how many rows are in it. Run this when diagnose_chat_session.py
can't find the table under a guessed schema name.

Usage:
    DATABASE_URL=<db-url> python scripts/list_chat_schemas.py
"""

import os
import sys

import psycopg2
import psycopg2.extras


def main() -> None:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    dsn = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )

    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """SELECT table_schema, table_name
           FROM information_schema.tables
           WHERE table_name IN ('chat_sessions', 'chat_messages')
           ORDER BY table_schema, table_name"""
    )
    rows = cur.fetchall()
    if not rows:
        print("No 'chat_sessions' or 'chat_messages' table found in ANY schema "
              "in this database. Either this is genuinely not the DB the app "
              "writes to, or migrations have never run against it at all.")
    else:
        print("Found chat tables in these schemas:")
        for r in rows:
            cur.execute(f'SELECT COUNT(*) AS n FROM "{r["table_schema"]}"."{r["table_name"]}"')
            n = cur.fetchone()["n"]
            print(f"  {r['table_schema']}.{r['table_name']}  ({n} rows)")

    print("\nAll non-system schemas in this database:")
    cur.execute(
        """SELECT schema_name FROM information_schema.schemata
           WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
           AND schema_name NOT LIKE 'pg_%'
           ORDER BY schema_name"""
    )
    for r in cur.fetchall():
        print(f"  {r['schema_name']}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
