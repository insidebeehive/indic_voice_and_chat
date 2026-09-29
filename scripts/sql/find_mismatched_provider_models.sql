-- Read-only audit: tenants whose STORED provider/model (or chat-voice
-- provider/model) look mismatched -- a stale model left over from a
-- provider switch made before the two PATCH-merge fixes landed
-- (src/api/tenants.py's _merge_layer_fields; src/config_tenant.py's
-- merge_provider_config). Safe to run on prod: SELECT only, no writes.
--
-- Schema/table/column: voicebot.tenants.pipeline_config (JSON), confirmed
-- against src/models/tenant.py (Tenant.pipeline_config) and
-- src/models/database.py (default schema "voicebot", overridable via
-- VOX_DB_SCHEMA) -- adjust the schema prefix below if this environment set
-- VOX_DB_SCHEMA to something else.
--
-- Coverage: stt / llm / tts / chat_voice.tts, matching the four layers the
-- PATCH-merge fix touches. `realtime` is intentionally omitted -- there is
-- no platform-level realtime default to leak from, and RealtimeUpdateIn's
-- own fields (model/voice) are less mechanically checkable here (no fixed
-- model-name convention across s2s providers the way bulbul:/eleven_/
-- saaras: are for sarvam/elevenlabs).
--
-- Only flags providers whose model-id CONVENTION is well known and
-- unambiguous (elevenlabs -> eleven_*, sarvam TTS -> bulbul*, sarvam STT ->
-- saaras*, gemini -> gemini-*, anthropic -> claude-*). A NULL model is NOT
-- flagged -- that just means the adapter's own built-in default applies,
-- which is correct/expected, not a mismatch. Providers with no confident
-- single naming convention (groq, deepgram, azure, google, vllm/openai-
-- compat, indicf5) are left unchecked rather than risk false positives on a
-- legitimate custom model id.

WITH layers AS (
    SELECT id, slug, name,
           'tts'             AS layer,
           pipeline_config -> 'tts' ->> 'provider'                    AS provider,
           pipeline_config -> 'tts' ->> 'model'                       AS model
    FROM voicebot.tenants
    UNION ALL
    SELECT id, slug, name,
           'chat_voice.tts'  AS layer,
           pipeline_config -> 'chat_voice' -> 'tts' ->> 'provider'    AS provider,
           pipeline_config -> 'chat_voice' -> 'tts' ->> 'model'       AS model
    FROM voicebot.tenants
    UNION ALL
    SELECT id, slug, name,
           'llm'             AS layer,
           pipeline_config -> 'llm' ->> 'provider'                    AS provider,
           pipeline_config -> 'llm' ->> 'model'                       AS model
    FROM voicebot.tenants
    UNION ALL
    SELECT id, slug, name,
           'stt'             AS layer,
           pipeline_config -> 'stt' ->> 'provider'                    AS provider,
           pipeline_config -> 'stt' ->> 'model'                       AS model
    FROM voicebot.tenants
)
SELECT id, slug, name, layer, provider, model
FROM layers
WHERE provider IS NOT NULL
  AND model IS NOT NULL
  AND (
       (lower(provider) = 'elevenlabs'                    AND model NOT ILIKE 'eleven%')
    OR (lower(provider) = 'sarvam' AND layer IN ('tts', 'chat_voice.tts')
                                                            AND model NOT ILIKE 'bulbul%')
    OR (lower(provider) = 'sarvam' AND layer = 'stt'       AND model NOT ILIKE 'saaras%')
    OR (lower(provider) = 'gemini'                         AND model NOT ILIKE 'gemini-%')
    OR (lower(provider) = 'anthropic'                      AND model NOT ILIKE 'claude-%')
  )
ORDER BY slug, layer;
