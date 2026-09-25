-- Chat agent health, from chat_turn_metrics / chat_tool_metrics. Postgres, read-only.
--
-- Usage (a plain postgresql:// URL — psql rejects the app's postgresql+asyncpg:// form):
--   psql "postgresql://user:pass@host/db" -v tenant=t_5e02798175b644e6 -v days=7 -f scripts/sql/chat_turn_health.sql
--   psql "postgresql://user:pass@host/db" -v tenant=all -v days=7 -f scripts/sql/chat_turn_health.sql
-- Add -v schema=<name> if the app's tables are not in the "voicebot" schema.
--
-- Days are Asia/Kolkata calendar days. created_at is stored as naive UTC
-- (server_default now() on a GMT database), hence the double AT TIME ZONE.
--
-- What to watch, and what each rate means:
--   exhausted_pct       turns that hit the tool-round cap and needed the forced final call
--   retry_pct           turns whose reply was unusable (empty) and had to be regenerated
--   unusable_esc_pct    retry fired AND the turn escalated: the "not able to answer, let me
--                       connect you" path. Before 18b7523 these never actually handed off.
--   escalated_pct       all turns that escalated, for any reason
--   tool_fail_pct       turns with at least one failed tool call
--   guard_*_pct         a guard replaced or downgraded the reply
--   directive_pct       the tool-failure directive fired (a CRM tool kept failing)
--   cache_hit_pct       cached input tokens / all input tokens

\if :{?tenant}
\else
  \set tenant all
\endif
\if :{?days}
\else
  \set days 7
\endif
\if :{?schema}
\else
  \set schema voicebot
\endif
SET search_path TO :"schema", public;

\echo '== Daily turn health =='
WITH t AS (
    SELECT *,
           (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata')::date AS day
    FROM chat_turn_metrics
    WHERE created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
      AND (:'tenant' = 'all' OR tenant_id = :'tenant')
)
SELECT
    day,
    count(*)                                                        AS turns,
    count(DISTINCT session_id)                                      AS sessions,
    round(count(*)::numeric / nullif(count(DISTINCT session_id), 0), 2) AS turns_per_session,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_ms)::int     AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms)::int     AS p95_ms,
    round(avg(llm_calls), 2)                                        AS avg_llm_calls,
    round(100.0 * avg(rounds_exhausted::int), 1)                    AS exhausted_pct,
    round(100.0 * avg(retry_fired::int), 1)                         AS retry_pct,
    round(100.0 * avg((retry_fired AND escalated)::int), 1)         AS unusable_esc_pct,
    round(100.0 * avg(escalated::int), 1)                           AS escalated_pct,
    round(100.0 * avg((tool_failures > 0)::int), 1)                 AS tool_fail_pct,
    round(100.0 * avg((tool_timeouts > 0)::int), 1)                 AS tool_timeout_pct,
    round(100.0 * avg(guard_hallucination_fired::int), 1)           AS guard_halluc_pct,
    round(100.0 * avg(guard_no_grounding_fired::int), 1)            AS guard_nogrnd_pct,
    round(100.0 * avg(guard_unverified_data_fired::int), 1)         AS guard_unverif_pct,
    round(100.0 * avg(failure_directive_fired::int), 1)             AS directive_pct,
    round(avg(reply_words), 1)                                      AS avg_reply_words,
    round(100.0 * sum(cached_tokens) / nullif(sum(input_tokens), 0), 1) AS cache_hit_pct
FROM t
GROUP BY day
ORDER BY day DESC;

\echo '== Tools: calls, failure rate and latency per tool (whole window) =='
SELECT
    m.tool_name,
    m.kind,
    count(*)                                                        AS calls,
    round(100.0 * avg((m.outcome <> 'ok')::int), 1)                 AS not_ok_pct,
    string_agg(DISTINCT m.outcome, ',')                             AS outcomes_seen,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY m.latency_ms)::int AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY m.latency_ms)::int AS p95_ms
FROM chat_tool_metrics m
WHERE m.created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
  AND (:'tenant' = 'all' OR m.tenant_id = :'tenant')
GROUP BY m.tool_name, m.kind
ORDER BY not_ok_pct DESC, calls DESC;

\echo '== Tool rounds per turn: how often the model spends rounds one tool at a time =='
-- Turns that made tool calls, by rounds used and by distinct tools called in one
-- round. After 0f5013c (batched lookups), max_tools_in_a_round > 1 should grow and
-- exhausted turns should shrink.
WITH per_round AS (
    SELECT m.turn_id, m.round_index, count(*) AS tools_in_round
    FROM chat_tool_metrics m
    WHERE m.created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
      AND (:'tenant' = 'all' OR m.tenant_id = :'tenant')
    GROUP BY m.turn_id, m.round_index
), per_turn AS (
    SELECT turn_id, count(*) AS rounds_with_tools, max(tools_in_round) AS max_tools_in_a_round
    FROM per_round GROUP BY turn_id
)
SELECT rounds_with_tools, max_tools_in_a_round, count(*) AS turns
FROM per_turn
GROUP BY rounds_with_tools, max_tools_in_a_round
ORDER BY rounds_with_tools, max_tools_in_a_round;

\echo '== Sessions that hit the unusable-reply escalation more than once =='
SELECT session_id,
       count(*) FILTER (WHERE retry_fired AND escalated) AS unusable_escalations,
       count(*)                                          AS turns,
       min(created_at)                                   AS first_turn_utc
FROM chat_turn_metrics
WHERE created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
  AND (:'tenant' = 'all' OR tenant_id = :'tenant')
GROUP BY session_id
HAVING count(*) FILTER (WHERE retry_fired AND escalated) > 1
ORDER BY unusable_escalations DESC, first_turn_utc DESC
LIMIT 25;
