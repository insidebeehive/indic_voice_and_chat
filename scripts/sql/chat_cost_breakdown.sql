-- Where chat spend goes, from chat_turn_metrics / chat_tool_metrics /
-- chat_sessions / embedding_usage / provider_costs. Postgres, read-only.
--
-- Usage (a plain postgresql:// URL — psql rejects the app's postgresql+asyncpg:// form):
--   psql "postgresql://user:pass@host/db" -v tenant=all -v days=14 -f scripts/sql/chat_cost_breakdown.sql
--   psql "postgresql://user:pass@host/db" -v tenant=t_5e02798175b644e6 -v days=14 -f scripts/sql/chat_cost_breakdown.sql
-- Add -v schema=<name> if the app's tables are not in the "voicebot" schema.
--
-- LLM cost per turn is recomputed from the turn's token counts and the
-- provider_costs rates (USD per 1K tokens), the same formula as
-- src/api/chat_cost.py: uncached input at the input rate, cached input at the
-- cached rate (input rate if none is set), output at the output rate.
-- "other" in section 1 is session cost not explained by turns: mostly
-- summarize_session calls and inbound voice-note transcription outside a turn.

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

CREATE TEMP VIEW cost_turns AS
WITH r AS (
    SELECT provider, model,
           cost_per_1k_input_tokens  AS in_rate,
           cost_per_1k_output_tokens AS out_rate,
           cost_per_1k_cached_tokens AS cached_rate
    FROM provider_costs WHERE kind = 'llm'
)
SELECT m.*,
       ((m.input_tokens - m.cached_tokens) * COALESCE(r1.in_rate, r2.in_rate, 0)) / 1000.0 AS uncached_in_cost,
       (m.cached_tokens * COALESCE(
            CASE WHEN r1.provider IS NOT NULL THEN COALESCE(r1.cached_rate, r1.in_rate)
                 ELSE COALESCE(r2.cached_rate, r2.in_rate) END, 0)) / 1000.0     AS cached_in_cost,
       (m.output_tokens * COALESCE(r1.out_rate, r2.out_rate, 0)) / 1000.0        AS out_cost
FROM chat_turn_metrics m
LEFT JOIN r r1 ON r1.provider = m.llm_provider AND r1.model = m.llm_model
LEFT JOIN r r2 ON r2.provider = m.llm_provider AND r2.model = ''
WHERE m.created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
  AND (:'tenant' = 'all' OR m.tenant_id = :'tenant');

CREATE TEMP VIEW cost_turns_c AS
SELECT *, uncached_in_cost + cached_in_cost + out_cost AS llm_cost FROM cost_turns;

\echo '== 1. Where the chat money goes (per tenant, USD) =='
WITH t AS (
    SELECT tenant_id,
           count(*)                         AS turns,
           count(DISTINCT session_id)       AS sessions,
           sum(llm_cost)                    AS llm,
           sum(COALESCE(tts_cost, 0))       AS voice_note_tts,
           sum(COALESCE(stt_cost, 0))       AS voice_note_stt
    FROM cost_turns_c GROUP BY tenant_id
), s AS (
    SELECT tenant_id, sum(cost) AS session_total
    FROM chat_sessions
    WHERE started_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
      AND (:'tenant' = 'all' OR tenant_id = :'tenant')
    GROUP BY tenant_id
), e AS (
    SELECT tenant_id,
           sum(cost) FILTER (WHERE purpose = 'search') AS kb_search_embed,
           sum(cost) FILTER (WHERE purpose = 'ingest') AS kb_ingest_embed
    FROM embedding_usage
    WHERE created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
      AND tenant_id IS NOT NULL
      AND (:'tenant' = 'all' OR tenant_id = :'tenant')
    GROUP BY tenant_id
)
SELECT t.tenant_id, t.sessions, t.turns,
       round(t.llm::numeric, 4)                                   AS llm_usd,
       round(t.voice_note_tts::numeric, 4)                        AS tts_usd,
       round(t.voice_note_stt::numeric, 4)                        AS stt_usd,
       round(COALESCE(e.kb_search_embed, 0)::numeric, 5)          AS kb_search_usd,
       round(COALESCE(e.kb_ingest_embed, 0)::numeric, 5)          AS kb_ingest_usd,
       round(COALESCE(s.session_total, 0)::numeric, 4)            AS session_cost_usd,
       round((COALESCE(s.session_total, 0) - t.llm - t.voice_note_tts - t.voice_note_stt)::numeric, 4)
                                                                  AS other_usd,
       round((t.llm / nullif(t.turns, 0))::numeric, 5)            AS llm_usd_per_turn,
       round((COALESCE(s.session_total, 0) / nullif(t.sessions, 0))::numeric, 4) AS usd_per_session
FROM t LEFT JOIN s USING (tenant_id) LEFT JOIN e USING (tenant_id)
ORDER BY t.llm DESC;

\echo '== 2. What the LLM money is spent on (whole window) =='
SELECT count(*)                                                         AS turns,
       round(avg(llm_calls), 2)                                         AS avg_llm_calls,
       round(sum(input_tokens)::numeric / nullif(sum(llm_calls), 0))    AS avg_input_tok_per_call,
       round(sum(cached_tokens)::numeric / nullif(sum(llm_calls), 0))   AS avg_cached_tok_per_call,
       round(sum(output_tokens)::numeric / nullif(sum(llm_calls), 0))   AS avg_output_tok_per_call,
       round(100.0 * sum(cached_tokens) / nullif(sum(input_tokens), 0), 1) AS cache_hit_pct,
       round((100.0 * sum(uncached_in_cost) / nullif(sum(llm_cost), 0))::numeric, 1) AS uncached_input_pct,
       round((100.0 * sum(cached_in_cost)   / nullif(sum(llm_cost), 0))::numeric, 1) AS cached_input_pct,
       round((100.0 * sum(out_cost)         / nullif(sum(llm_cost), 0))::numeric, 1) AS output_pct,
       round(sum(llm_cost)::numeric, 4)                                 AS llm_usd
FROM cost_turns_c;

\echo '== 3. Cost by number of tool rounds =='
SELECT CASE WHEN rounds_exhausted THEN 'exhausted (forced final)' ELSE rounds::text END AS rounds,
       count(*)                                                       AS turns,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)             AS turns_pct,
       round(avg(llm_calls), 2)                                       AS avg_llm_calls,
       round(avg(llm_cost)::numeric, 5)                               AS avg_usd,
       round((100.0 * sum(llm_cost) / sum(sum(llm_cost)) OVER ())::numeric, 1) AS cost_pct
FROM cost_turns_c
GROUP BY 1 ORDER BY 1;

\echo '== 4. Cost by the first tool the turn called =='
WITH first_tool AS (
    SELECT DISTINCT ON (turn_id) turn_id, tool_name
    FROM chat_tool_metrics
    WHERE created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
    ORDER BY turn_id, round_index, id
)
SELECT COALESCE(f.tool_name, '(no tool)')                             AS first_tool,
       count(*)                                                       AS turns,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)             AS turns_pct,
       round(avg(t.llm_calls), 2)                                     AS avg_llm_calls,
       round(avg(t.tool_calls), 2)                                    AS avg_tool_calls,
       round(avg(t.llm_cost)::numeric, 5)                             AS avg_usd,
       round((100.0 * sum(t.llm_cost) / sum(sum(t.llm_cost)) OVER ())::numeric, 1) AS cost_pct
FROM cost_turns_c t LEFT JOIN first_tool f ON f.turn_id = t.id
GROUP BY 1 ORDER BY sum(t.llm_cost) DESC;

\echo '== 5. Extra calls: unusable-reply retries and forced final answers =='
SELECT retry_fired, rounds_exhausted,
       count(*)                                                       AS turns,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)             AS turns_pct,
       round(avg(llm_calls), 2)                                       AS avg_llm_calls,
       round(avg(llm_cost)::numeric, 5)                               AS avg_usd,
       round((100.0 * sum(llm_cost) / sum(sum(llm_cost)) OVER ())::numeric, 1) AS cost_pct
FROM cost_turns_c
GROUP BY 1, 2 ORDER BY 1, 2;

\echo '== 6. Sessions: turns and LLM cost per session =='
WITH s AS (
    SELECT session_id, count(*) AS turns, sum(llm_cost) AS usd, bool_or(escalated) AS escalated
    FROM cost_turns_c GROUP BY session_id
)
SELECT CASE WHEN turns >= 10 THEN '10+' WHEN turns >= 5 THEN '5-9' ELSE turns::text END AS turns_per_session,
       escalated,
       count(*)                                                       AS sessions,
       round(avg(usd)::numeric, 4)                                    AS avg_usd,
       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY usd)::numeric, 4) AS p50_usd,
       round(percentile_cont(0.9) WITHIN GROUP (ORDER BY usd)::numeric, 4) AS p90_usd,
       round((100.0 * sum(usd) / sum(sum(usd)) OVER ())::numeric, 1)  AS cost_pct
FROM s
GROUP BY 1, 2 ORDER BY min(turns), 2;

\echo '== 7. Tool results re-sent to the model (size per call) =='
SELECT tool_name, kind,
       count(*)                                                       AS calls,
       round(avg(result_chars))                                       AS avg_result_chars,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY result_chars)::int AS p95_result_chars,
       round(avg(round_index), 2)                                     AS avg_round
FROM chat_tool_metrics
WHERE created_at >= (now() AT TIME ZONE 'UTC') - make_interval(days => :'days'::int)
  AND (:'tenant' = 'all' OR tenant_id = :'tenant')
  AND result_chars IS NOT NULL
GROUP BY tool_name, kind
ORDER BY sum(result_chars) DESC;

\echo '== 8. Daily trend (Asia/Kolkata days) =='
SELECT (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Kolkata')::date AS day,
       count(*)                                                       AS turns,
       round(avg(llm_calls), 2)                                       AS avg_llm_calls,
       round(100.0 * sum(cached_tokens) / nullif(sum(input_tokens), 0), 1) AS cache_hit_pct,
       round(sum(llm_cost)::numeric, 4)                               AS llm_usd,
       round((sum(llm_cost) / nullif(count(*), 0))::numeric, 5)       AS usd_per_turn,
       round(sum(COALESCE(tts_cost, 0) + COALESCE(stt_cost, 0))::numeric, 4) AS voice_note_usd
FROM cost_turns_c
GROUP BY 1 ORDER BY 1 DESC;
