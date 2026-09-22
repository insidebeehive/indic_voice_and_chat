# Session Handoff — Barge-in (HISTORICAL SNAPSHOT, 2026-06-04)

> **Do not act on this document.** It is a point-in-time snapshot of one
> session's working tree, kept for the measurements in it, and the mechanism it
> describes no longer exists.
>
> Barge-in detection is **server-side**, not browser-side: `_barge_on_interim()`
> in `src/api/browser_bridge.py`, triggering on sustained interim STT past
> `BARGE_SUSTAIN_MS` (450ms). The client-side RMS detector this document tunes
> — `BARGE_RMS`, `bargeMs`, the worklet VAD branch — was deliberately removed
> (`docs/superpowers/specs/2026-06-10-barge-in-dev-console-design.md`), so none
> of those identifiers exist in the tree.
>
> It also says to re-add temporary `[barge] rms=…` diagnostics when debugging.
> That is no longer necessary: barge-in fire and suppress paths are permanently
> instrumented in `browser_bridge.py`, `live_bridge_base.py`,
> `telephony_live_bridge.py` and `livekit_bridge.py`, each carrying what was in
> flight and what was cancelled. Turn on `VOX_LOG_LEVEL=DEBUG` instead — see
> `docs/debug-logging.md`.
>
> What remains useful below is the measured data: user speech at ~0.008–0.012
> RMS post-AEC against an echo floor under 0.005, and Deepgram's 1000ms
> `UtteranceEnd` post-gap floor.

**Date:** 2026-06-04. Captures the uncommitted working-tree state of that session.

---

## TL;DR — what to do on resume

Barge-in is **built, reviewed, working, and the working tree is CLEAN** (DIAG stripped, tuning committed). We're at the **final validation** step. The user confirmed barge-in fires and addresses the interruption; the earlier stuck/slow-after-interruption bug was root-caused and **fixed** (commit `6bc0ce4` — stale Deepgram `_endpointed` flag). **Still awaiting the user's live retest of that fix** (they hit an unrelated ngrok outage before retesting).

**Immediate next steps:**
1. Relaunch the server (it'll be down after a machine restart — command in Operational below).
2. Ask the user to retest barge-in at **http://localhost:8765/dev/voice** (NOT ngrok) and confirm: no more stuck, delay-before-thinking acceptable.
3. If good → **push** (origin/main is ~11 commits behind; user wanted to confirm before pushing) and optionally fold the barge-in outcome into `docs/latency-llm-stt-experiments.md`. That closes the feature.
4. If still self-triggering on echo → raise `BARGE_RMS` toward 0.007 in `static/dev_console.html`. If still missing interruptions → lower toward 0.005. If still slow-but-not-stuck → it's Deepgram falling back to the 1000ms `UtteranceEnd` post-gap (its minimum; can't lower) — discuss keeping the stream warm during agent turns as a follow-up. (If you need to debug again, re-add diagnostics — the prior DIAG approach: log `[barge] rms=...` in the worklet + `barge_in frame received`/`agent_busy` in the bridge.)

---

## Working-tree state — CLEAN (as of `7ea8d18`)

`git status` is clean except untracked `.claude/` (never commit). The DIAG logging has been **stripped** and the validated barge-in tuning **committed** (`7ea8d18`): `BARGE_RMS=0.006` (was 0.02 — measured user speech ~0.008–0.012 RMS post-AEC, echo floor <0.005) + a decay accumulator (`Math.max(0, bargeMs - frameMs)` instead of hard reset). `src/api/browser_bridge.py` is back to its committed form. Nothing to strip or recover.

> `src/providers/stt/deepgram.py`'s `log.warning("deepgram stream ended unexpectedly", ...)` is committed observability (931c277), not DIAG — leave it.

---

## Barge-in feature status (the in-flight work)

- **Spec:** `docs/superpowers/specs/2026-06-04-barge-in-design.md` (committed `dceb70a`).
- **Plan:** `docs/superpowers/plans/2026-06-04-barge-in.md` (committed `e609a56`).
- **Approach:** browser-side detection (VAD on echo-cancelled mic) + instant `stopPlayback()` + `{"type":"barge_in"}` → server `_handle_barge_in()` sets the in-flight turn's `cancel_event` + clears `_agent_busy` → engine stops LLM/TTS → `handle_turn_text` records the user turn, drops the abandoned reply, → LISTENING → interruption becomes the next turn. The cancel/turn-handling core is transport-agnostic (telephony reuses it later via a server-side detector calling the same `_handle_barge_in()`).
- **Implemented & reviewed (subagent-driven, all on `main`):**
  - `cfbae82` engine cancel-stops-audio test.
  - `db657e1` voicebot: `handle_turn_text(cancel_event=)`; cancelled turn keeps user turn, drops reply, → LISTENING (`parse_error="barge-in"`).
  - `3360de2` bridge: `_handle_barge_in()`, per-turn `cancel_event` in `_dispatch_text_turn`, `barge_in` routing.
  - `c7e4fe4` browser: `activeSources`/`stopPlayback()`, VAD detect, "Allow interruptions" toggle.
  - Final holistic review: **Ready to merge.** 627 tests passed at that point.
- **Bug found in live test + FIXED:** `6bc0ce4` — after a `speech_final`, Deepgram session's `_endpointed` stayed True and suppressed the next utterance's `UtteranceEnd` backup → the post-barge-in interruption could get **stuck in listening** (or slow via the 1000ms UtteranceEnd). Fix clears `_endpointed` when new speech arrives. Has a regression test. 628 tests pass.
- **Remaining:** confirm the fix live → strip DIAG → commit tuned `dev_console.html` → push.

---

## Git state

- **Local `main` HEAD:** `5132cb7` (server-arms barge-in only during a cancellable turn — fixes the greeting self-cutoff: on speakers the VAD fired on the opening greeting's echo and `stopPlayback()` cut it; server now sends `{type:barge,armed:true/false}` and the browser only detects while armed). Preceded by `7ea8d18` (VAD tuning), `8dc0d7e` (handoff update), `6bc0ce4` (deepgram endpoint fix).
- **Unpushed:** everything after `8dc0d7e` (which was the last push). i.e. `5132cb7` is unpushed.
- **`origin/main`:** behind — these commits are **NOT pushed yet**: `c630d46` (turn-timeout fix), `931c277` (streaming-STT crash/drop logging), `dceb70a`+`e609a56` (barge-in spec+plan), `cfbae82` `db657e1` `3360de2` `c7e4fe4` (barge-in impl), `6bc0ce4` (deepgram endpoint fix), plus the docs commits from earlier (`3476f51`, `49a6a25` were pushed; verify). **Push after barge-in is finalized** (user has been asked to confirm before pushing).
- Untracked `.claude/` — never commit.

---

## Operational (how to run / test)

- **Run the dev server:**
  `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`
  (The repo's venv is `.venv`; tests/run use it. `anthropic`, `deepgram-sdk` are installed there and pinned in `pyproject.toml`.)
- **Test the console in the browser:** **http://localhost:8765/dev/voice** — use localhost, NOT ngrok. `getUserMedia` treats localhost as a secure context, so the mic works; no tunnel hop = lower latency. ngrok is only needed for telephony webhooks.
- **ngrok is currently DOWN** (ERR_NGROK_3200). Only restart it if doing a real phone call:
  `ngrok http --url=conceitedly-sinewy-margurite.ngrok-free.dev 8765`
- **Active config (`config/tenants/dev.yaml`):** LLM = `gemini-2.5-flash`; STT batch = Groq; **STT streaming = Deepgram `nova-2 hi`, `endpointing=300`, `utterance_end_ms=1000`** (the dev console uses streaming). Keys in gitignored `.env`: `TENANT_DEV_{GEMINI,ANTHROPIC,DEEPGRAM,GROQ,SARVAM}_KEY`.
- **Per-turn metrics** in the server log: `"browser turn (stream)"` lines carry `llm_ttft_ms/llm_total_ms/tts_first_ms/total_ms/action/user_text/agent_text`. Barge-in logs `barge-in: cancelling current turn`.
- **Note:** the dev server is currently running (PID was 30523); after a machine/session restart it will be down — relaunch with the command above.

---

## Bigger-picture context (already committed/documented)

- **Decision log** (Claude-vs-Gemini, cloud latency, Deepgram streaming, bottleneck = LLM): `docs/latency-llm-stt-experiments.md`; index at `docs/README.md`.
- **Deepgram streaming STT** feature: shipped + pushed earlier (interface `src/interfaces/stt.py`, adapter `src/providers/stt/deepgram.py`, registry `STREAMING_STT_PROVIDERS`, engine `run_turn_text`, bridge streaming path). Kept ON as the dev-console default; quality/reliability win, ~0.7s off the pre-response gap.
- **Turn-timeout safety net** (`c630d46`): every turn bounded by `TURN_TIMEOUT_S=20.0` (`src/agents/voicebot.py`) so a hung provider can't wedge the agent.
- **Known limitations / standing decisions:** kept Gemini (Claude was a latency wash + had a 36s stall; Claude adapter available behind `provider: anthropic|claude`). Latency floor is LLM-inherent (~2.0–2.5s to first word). Don't leave the dev server running for days — a stale Gemini client caused a latency creep that a restart fixed.
