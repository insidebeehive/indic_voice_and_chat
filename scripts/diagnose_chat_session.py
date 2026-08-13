"""Find a chat session by customer name + approximate time (when you only
have a CRM ticket ID, which our DB doesn't store) and dump its full
persisted transcript, so you can tell whether the bot ever actually
replied — vs. the reply existing but not being rendered by the widget.

Usage:
    DATABASE_URL=<db-url> python scripts/diagnose_chat_session.py \
        --name "Sajid Vhora" --date 2026-08-13 --around 20:45 --window-min 15
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Customer name, partial match ok")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--around", required=True, help="HH:MM (24h, session's local/UTC time as stored)")
    parser.add_argument("--window-min", type=int, default=15, help="Minutes of slack either side (default 15)")
    parser.add_argument("--schema", default=os.environ.get("VOX_DB_SCHEMA", "voicebot"),
                         help="Postgres schema our tables live under (default: voicebot, "
                              "or $VOX_DB_SCHEMA if set)")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    dsn = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )

    center = datetime.strptime(f"{args.date} {args.around}", "%Y-%m-%d %H:%M")
    lo = center - timedelta(minutes=args.window_min)
    hi = center + timedelta(minutes=args.window_min)

    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # Tables live under a non-default schema (src/models/database.py), same as
    # the app pins via asyncpg server_settings — mirror it here for psycopg2.
    # SET doesn't take bind params for identifiers, so quote it ourselves
    # (schema name here is an operator-supplied CLI arg, not user input).
    quoted_schema = '"' + args.schema.replace('"', '""') + '"'
    cur.execute(f'SET search_path TO {quoted_schema}, public')

    cur.execute(
        """SELECT id, tenant_id, customer_name, status, mode, language,
                  message_count, cost, started_at, ended_at
           FROM chat_sessions
           WHERE customer_name ILIKE %s AND started_at BETWEEN %s AND %s
           ORDER BY started_at""",
        (f"%{args.name}%", lo, hi),
    )
    sessions = cur.fetchall()
    if not sessions:
        print(f"No sessions found for name~='{args.name}' between {lo} and {hi}.")
        print("Try a wider --window-min, or check the name spelling (CRM display name "
              "may differ from what was passed at session creation).")
        return

    for s in sessions:
        print(f"\n=== session {s['id']} ===")
        print(f"  tenant={s['tenant_id']} status={s['status']} mode={s['mode']} "
              f"lang={s['language']} messages={s['message_count']} cost={s['cost']}")
        print(f"  started_at={s['started_at']} ended_at={s['ended_at']}")

        cur.execute(
            """SELECT id, role, type, content, media_mime, llm_provider, llm_model,
                      input_tokens, output_tokens, cost, created_at
               FROM chat_messages WHERE session_id = %s ORDER BY id""",
            (s["id"],),
        )
        msgs = cur.fetchall()
        if not msgs:
            print("  -- ZERO messages persisted for this session --")
            print("  This means either (a) the customer's message never reached the "
                  "server at all, or (b) it reached the server but a failure happened "
                  "before _persist_turn ever ran (check for a factory-construction or "
                  "early-exception failure, not a per-turn one).")
            continue

        agent_replies = [m for m in msgs if m["role"] == "agent"]
        print(f"  -- {len(msgs)} message(s), {len(agent_replies)} agent repl(y/ies) --")
        for m in msgs:
            preview = (m["content"] or "")[:200].replace("\n", " ")
            extra = ""
            if m["role"] == "agent" and m["llm_provider"]:
                extra = f" [llm={m['llm_provider']}/{m['llm_model']} in={m['input_tokens']} out={m['output_tokens']} cost={m['cost']}]"
            print(f"    #{m['id']} {m['created_at']} {m['role']:9s} {m['type']:6s} {preview!r}{extra}")

        if not agent_replies:
            print("  => Customer message(s) WERE persisted, but the bot NEVER replied.")
            print("     This points at a per-turn failure that's silently swallowing "
                  "the error rather than sending an error frame, OR at agent.handle_message()/"
                  "handle_image() itself hanging/failing before _run_turn returns.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
