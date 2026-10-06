-- Customer chat messages, each matched to the agent turn that answered it,
-- as CSV for scripts/analyze_customer_queries.py. Postgres, read-only.
--
-- Usage (a plain postgresql:// URL — psql rejects the app's postgresql+asyncpg:// form):
--   psql "postgresql://user:pass@host/db" -v tenant=all -v days=14 -q \
--        -f scripts/sql/export_customer_queries.sql > queries.csv
-- Add -v schema=<name> if the app's tables are not in the "voicebot" schema.
--
-- The CSV holds raw customer text: keep it on your machine and delete it
-- after the analysis. A message is matched to the turn metric in the same
-- session closest in time (within 3 minutes); a message with no turn nearby
-- (e.g. sent while a human agent had the chat) has empty turn columns.

-- Quiet mode: without it psql writes "SET" ahead of the CSV header.
\set QUIET on

\if :{?tenant}
\else
  \set tenant all
\endif
\if :{?days}
\else
  \set days 14
\endif
\if :{?schema}
\else
  \set schema voicebot
\endif
SET search_path TO :"schema", public;

COPY (
    WITH msgs AS (
        SELECT m.session_id, m.created_at, m.content,
               row_number() OVER (PARTITION BY m.session_id ORDER BY m.created_at, m.id) = 1 AS first_msg
        FROM chat_messages m
        JOIN chat_sessions s ON s.id = m.session_id
        WHERE m.role = 'customer'
          AND m.created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
          AND (:'tenant' = 'all' OR s.tenant_id = :'tenant')
    )
    SELECT ms.session_id, ms.created_at, ms.content, ms.first_msg,
           t.llm_calls, t.input_tokens, t.cached_tokens, t.output_tokens,
           ft.tool_name AS first_tool
    FROM msgs ms
    LEFT JOIN LATERAL (
        SELECT tm.id, tm.llm_calls, tm.input_tokens, tm.cached_tokens, tm.output_tokens
        FROM chat_turn_metrics tm
        WHERE tm.session_id = ms.session_id
          AND tm.created_at BETWEEN ms.created_at - interval '3 minutes'
                                AND ms.created_at + interval '3 minutes'
        ORDER BY abs(extract(epoch FROM tm.created_at - ms.created_at))
        LIMIT 1
    ) t ON true
    LEFT JOIN LATERAL (
        SELECT x.tool_name FROM chat_tool_metrics x
        WHERE x.turn_id = t.id
        ORDER BY x.round_index, x.id
        LIMIT 1
    ) ft ON true
    ORDER BY ms.created_at
) TO STDOUT WITH CSV HEADER;
