-- Read-only audit, run before deploying the change that makes chat voice-note
-- replies pass the tenant's configured voice_id to the TTS provider.
-- Safe on prod: SELECT only, no writes.
--
-- Before that change, Sarvam/Google/Azure/Gemini voice-note replies ignored
-- the configured voice_id and spoke the adapter's built-in default, so a
-- stale voice id never failed. Now it reaches the provider, and an invalid
-- one (e.g. a Sarvam bulbul:v2 speaker such as meera/anushka/arjun under
-- bulbul:v3) makes synthesis fail and the reply goes out text-only.
--
-- Lists every chat-voice-enabled tenant's effective chat TTS layer
-- (chat_voice.tts when it declares a provider, else pipeline.tts -- the same
-- rule as src/config_tenant.py resolve_chat_tts_config), with voice_id and
-- the per-gender voices pair. `sarvam_v3_ok` checks voice_id and both
-- voices entries of Sarvam tenants against the bulbul:v3 roster in
-- src/providers/tts/sarvam.py (_BULBUL_V3_SPEAKERS);
-- it is NULL for other providers, so check those by eye.
--
-- Schema: voicebot (see find_mismatched_provider_models.sql for the
-- VOX_DB_SCHEMA note).

WITH roster(v) AS (
    VALUES ('aditya'), ('ritu'), ('ashutosh'), ('priya'), ('neha'), ('rahul'),
           ('pooja'), ('rohan'), ('simran'), ('kavya'), ('amit'), ('dev'),
           ('ishita'), ('shreya'), ('ratan'), ('varun'), ('manan'), ('sumit'),
           ('roopa'), ('kabir'), ('aayan'), ('shubh'), ('advait'), ('anand'),
           ('tanya'), ('tarun'), ('sunny'), ('mani'), ('gokul'), ('vijay'),
           ('shruti'), ('suhani'), ('mohit'), ('kavitha'), ('rehan'),
           ('soham'), ('rupali')
),
effective AS (
    SELECT id, slug, name,
           CASE WHEN NULLIF(pipeline_config -> 'chat_voice' -> 'tts' ->> 'provider', '') IS NOT NULL
                THEN 'chat_voice.tts' ELSE 'tts' END AS layer,
           CASE WHEN NULLIF(pipeline_config -> 'chat_voice' -> 'tts' ->> 'provider', '') IS NOT NULL
                THEN pipeline_config -> 'chat_voice' -> 'tts'
                ELSE pipeline_config -> 'tts' END AS cfg
    FROM voicebot.tenants
    WHERE (pipeline_config -> 'chat_voice' ->> 'enabled') = 'true'
)
SELECT id, slug, name, layer,
       cfg ->> 'provider'                AS provider,
       cfg ->> 'model'                   AS model,
       cfg ->> 'voice_id'                AS voice_id,
       cfg -> 'voices' ->> 'female'      AS voice_female,
       cfg -> 'voices' ->> 'male'        AS voice_male,
       CASE WHEN lower(cfg ->> 'provider') = 'sarvam'
            THEN (cfg ->> 'voice_id' IS NULL
                  OR lower(cfg ->> 'voice_id') IN (SELECT v FROM roster))
             AND (cfg -> 'voices' ->> 'female' IS NULL
                  OR lower(cfg -> 'voices' ->> 'female') IN (SELECT v FROM roster))
             AND (cfg -> 'voices' ->> 'male' IS NULL
                  OR lower(cfg -> 'voices' ->> 'male') IN (SELECT v FROM roster))
       END                               AS sarvam_v3_ok
FROM effective
WHERE NULLIF(cfg ->> 'provider', '') IS NOT NULL
ORDER BY sarvam_v3_ok NULLS LAST, slug;
