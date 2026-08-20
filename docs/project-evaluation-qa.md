# Project Evaluation: Likely Technical Questions and Answers

Grounded in `docs/chatbot-report.md`, `docs/voice-pipeline-benchmarking-report.md`,
`docs/VOICE-ARCHITECTURE.md`, and `docs/PROJECT-STATUS.md`. These are questions an
examiner would actually ask given what's *in* the codebase and reports — not generic
ML-project boilerplate.

## A. Voice pipeline latency benchmarking (`voice-pipeline-benchmarking-report.md`)

### Statistics methodology

1. **Why log-transform latency before ANOVA?**
   Raw latencies are right-skewed (slow outliers), violating ANOVA's
   normality-of-residuals assumption; `log(1+x)` is standard for timing data, means
   are back-transformed to ms for interpretability.

2. **Why Type II sum of squares for the factorial ANOVA, not Type I or III?**
   Type II is appropriate when testing main effects without assuming a specific
   term-entry order and without a significant higher-order interaction that would
   require Type III; be ready to justify this choice if pressed, since the report
   doesn't argue it explicitly.

3. **What does η²=0.61 for TTS actually mean, and why not just report the F-statistic?**
   η² gives proportion of variance explained (comparable across factors,
   sample-size-independent-ish), where F/p only tell you "significant or not," not
   "how much it matters."

4. **Why run both one-way ANOVA and a three-way factorial ANOVA rather than just the factorial model?**
   One-way is more robust given modest per-cell samples (9-24 turns); the factorial
   model additionally tests interaction terms the one-way approach can't detect.

5. **Why exclude three-way interactions (STT×LLM×TTS)?**
   Sample size doesn't support reliable estimation of a 3×3×4 three-way term — be
   ready to explain why (too many cells, too few turns per cell).

6. **Why two different tests (Welch's t-test + Mann-Whitney U) for the S2S comparison instead of one?**
   One parametric (robust to unequal variance), one non-parametric/distribution-free
   — the conclusion doesn't rest on either test's assumptions alone; both agreeing
   strengthens the claim.

7. **This isn't a randomized controlled experiment — how does that limit the conclusions?**
   Manual test calls, not randomized order, varying content/length/time-of-day;
   described as "descriptive and comparative," not causal-strength evidence. Expect
   a direct question: "so is this even a valid experiment?"

8. **Why is the S2S `tts_first_chunk_ms` measurement anchored differently from the cascade's, and why does that matter?**
   Anchored from last-detected caller speech (not turn-start); an earlier version
   anchored from turn-start and produced numbers that contradicted the subjectively
   fast feel of S2S — a live example of a measurement bug that would have flipped
   the finding.

### Substantive findings

9. **Why does TTS dominate `tts_first_chunk_ms` variance but STT/LLM don't?**
   Cascade architecture: TTS synthesis starts on the LLM's first sentence, largely
   independent of which STT/LLM produced the text upstream — so TTS is structurally
   the last stage gating first audio.

10. **Why does STT/LLM matter for `total_latency_ms` but not `tts_first_chunk_ms`?**
    Total latency accumulates all sequential stage time; first-chunk latency is
    gated by whichever stage produces the first audible output.

11. **What do the significant STT×TTS / LLM×TTS interaction terms mean practically?**
    Effects aren't purely additive; a specific combo should be spot-checked rather
    than assumed from independent averages, especially slow-TTS + slow-LLM pairings.

12. **Why is S2S 4x faster on first-response but not on total latency?**
    No discrete STT→LLM→TTS serialization for S2S, so it wins on *starting* to
    respond; but total latency includes full spoken-response duration, where S2S has
    no comparable advantage (reply playback still runs ~7-12s per
    `PROJECT-STATUS.md`).

13. **Given S2S is faster, why does the cascade remain the default?**
    Cascade is cheaper, more controllable (structured JSON envelope vs.
    tool-call-only state extraction), and better suited where cost/control matter
    more than raw first-word latency.

14. **How does S2S give you structured state (slots, actions) if the model only emits audio?**
    Via a `record_turn_signal(action, updated_slots)` tool call the model makes
    mid-speech, feeding the same `apply_signal`/state-machine/outcome-analysis path
    as the cascade's JSON envelope.

## B. ChatBot / RAG module (`chatbot-report.md`)

15. **Why two execution paths (`_single_shot` vs `_handle_with_tools`) instead of one?**
    Gemini's SDK makes structured-JSON output and native tool-calling mutually
    incompatible; tool mode requires plain-text responses across up to 2 rounds,
    with a forced final plain-text-only call to guarantee a reply.

16. **Why cap `max_tool_rounds` at 2?**
    Prevents an unbounded tool-calling loop; forces termination with a guaranteed
    reply rather than risking silent failure.

17. **Why rebuild the system prompt from scratch every turn instead of caching it?**
    Lets per-turn language directives and fresh retrieval context be injected
    precisely, at the cost of recomputation — deliberate trade-off, not an
    oversight.

18. **Why cap replayed history to the last 10 exchanges when the full transcript is retained?**
    Bounds latency/token cost growth (~2 entries/turn otherwise) while keeping the
    full transcript for UI/audit.

19. **Walk through the three checks in the hallucination guard and why each exists.**
    Citation validation (strip fabricated `sources_used`, downgrade confidence),
    zero-retrieval override (no chunks → forced fallback regardless of model's own
    confidence claim), silent-hallucination detection (high confidence + zero
    citations = suspicious, substituted). Be ready to explain why this can't just be
    a prompt instruction — it's enforced in code precisely because prompt
    instructions alone are not reliable enough for a "never fabricate balances"
    guarantee.

20. **Why does the guard skip turns with no retrieval attempted?**
    Greetings/image-only/CRM-tool-resolved turns are "legitimately ungrounded," not
    grounding failures — guarding them would produce false-positive fallbacks.

21. **Why hybrid dense+sparse (FAISS+BM25) retrieval instead of dense-only?**
    Dense embeddings catch semantic similarity; BM25 recovers exact terminology
    (product names, error codes) dense embeddings can under-weight — standard for
    terminology-heavy support docs.

22. **What's the Gemini-embedding normalization bug this project actually hit, and why is it dangerous?**
    Gemini's embedding API doesn't L2-normalize at reduced dimensionality; FAISS
    index is built for inner-product similarity, so un-normalized vectors silently
    degrade retrieval without raising an error — a good "describe a subtle bug you
    found" answer.

23. **Walk through the language-detection subsystem's 4-revision history and why each fix introduced a new failure.**
    Almost certainly asked directly, given the report frames it as the central case
    study (§4.3):
    - Script-detection → all-Latin-as-English (breaks Hinglish)
    - Latin-as-no-signal (breaks tenant-default-language override)
    - Advisory instruction (too weak against conversational momentum)
    - Deterministic marker-based classifier with firm override (fixes momentum,
      leaves short messages under-specified)
    - Opening-message-only Hinglish default (scoped narrowly to avoid reopening the
      momentum bug)

    Know *why* each fix broke the previous one, not just the sequence.

24. **Why does the final fix scope the Hinglish default to "session's first message only," and how is that guaranteed safe?**
    Checked via "has this session ever seen a user turn" — makes the code path
    structurally unreachable once one user turn exists, provably (not just
    conventionally) avoiding the third failure mode.

25. **Why is "script is not language" the core problem statement, and why can't you just check for Latin vs. non-Latin script?**
    Romanized Hindi is character-identical to English at the Latin-script level;
    only native-script text is unambiguous.

26. **Why exclude "do"/"the" etc. from the 40-word Hinglish marker list?**
    Avoids false-positive Hinglish classification on genuine English words that
    happen to double as Hindi transliterations.

27. **Why degrade gracefully (empty results / holding message) rather than raise on tool/retrieval failure?**
    Customer-facing system — an unhandled exception mid-conversation is worse than
    an honest, unhelpful holding message.

28. **Why is the Chatwoot inbound webhook unauthenticated, and is that a vulnerability?**
    Explicitly documented as an accepted trade-off (temporary integration) — know
    this cold since it reads as a glaring gap to a reviewer who doesn't have that
    context.

29. **Why is escalation-webhook signing fail-open (still sends unsigned if no secret configured) rather than fail-closed?**
    A missed escalation is judged worse than an unsigned payload to a tenant with no
    secret configured yet.

30. **Why is retrieval run only against caption/text on multimodal turns, not the image/video itself?**
    Retrieval pipeline is text-only by design (§3); images are passed inline to the
    LLM directly, not through RAG.

31. **The "200 of 200 sessions" load-test figure — is that current data?**
    No — the report explicitly flags this could not be reconfirmed against a
    specific run and should be treated as provisional pending a fresh logged run.
    (Prior project history suggests this WAS validated at some point — be ready to
    reconcile: the report is being conservative/honest about not having a fresh,
    reproducible log for that exact claim.)

32. **A stale unit test asserts prompt guardrail language that's since changed — is that a missing guardrail or a stale test?**
    Confirmed stale test, not missing guardrail; fix is a test update.

## C. Cross-cutting / architecture

33. **Two "brains," one dialogue manager — how do cascade and S2S share state despite producing output differently?**
    Both funnel into `VoiceBotAgent.apply_signal()` → state machine + slots +
    prompts + outcome analysis; they differ only in how speech↔intent is produced
    (serialized STT→LLM→TTS vs. end-to-end audio + tool call).

34. **Why is `PROJECT-STATUS.md` (dated 2026-06-15) saying RAG/ChatBot is an "untouched scaffold" when the chatbot report describes a fully built module?**
    This status doc predates the ChatBot build (PRs #119–127) and is stale on this
    point — an evaluator who reads both docs could catch this inconsistency, so
    it's worth updating `PROJECT-STATUS.md` or being ready to explain the timeline
    gap.

35. **Why are providers behind interfaces (`ISTTProvider`, `ITTSProvider`, etc.) rather than hard-coded?**
    Swappable-provider pattern used consistently across STT/LLM/TTS/vector-store/
    telephony — enables the 36-combination benchmarking matrix and per-tenant
    provider choice without code branching.

36. **Multi-tenant credential isolation — how do you guarantee tenant A's CRM tool call never uses tenant B's credentials?**
    Per-tenant credential resolution enforced even when falling back to a shared CRM
    tool catalog.

37. **Telephony barge-in is missing everywhere except the dev console and S2S — why, and what's the gap?**
    Twilio/Exotel's `handle_turn` has no `cancel_event` yet; Stringee only has a
    coarse SCCO flag, not real detection — documented as a known fast-follow, not an
    oversight.

## Known open item

`PROJECT-STATUS.md` (dated 2026-06-15) is stale relative to `chatbot-report.md` on
whether RAG/ChatBot is built — see Q34. Worth reconciling before an evaluator spots
it independently.
