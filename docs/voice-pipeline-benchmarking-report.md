# Voice Pipeline Latency Benchmarking: A Factorial Analysis of STT, LLM, and TTS Provider Choice

## Abstract

The platform's layered (cascade) voice pipeline composes three independently swappable providers — speech-to-text (STT), large language model (LLM), and text-to-speech (TTS) — behind a common provider-registry interface. This chapter presents a systematic, statistically-grounded comparison of every combination in the currently supported 3×3×4 provider matrix (three STT providers, three LLM providers, four TTS providers), using per-turn latency instrumentation collected from live test calls. A two-way analysis of variance (ANOVA), performed separately on each of two latency metrics, isolates the independent contribution of each provider dimension and quantifies which combinations of provider choices interact non-additively. The central finding is that TTS provider selection is overwhelmingly the dominant factor in perceived responsiveness — accounting for roughly 61% of the explainable variance in time-to-first-spoken-response — while STT and LLM choice have no measurable effect on that specific metric, but do materially affect total turn duration. A separate comparison against the platform's speech-to-speech (S2S) architecture — a single end-to-end audio model with no decomposable STT/LLM/TTS stages, and therefore analyzed descriptively rather than as a factor in the ANOVA — finds S2S to be significantly faster than even the best-performing layered combination, at roughly a quarter of its response latency.

## 1. Motivation and Scope

Earlier informal comparisons across a handful of provider combinations (documented separately in this project's latency/LLM/STT experiment log) established that LLM and TTS inference time dominate overall pipeline latency, and that combination testing had, until recently, not been systematically logged at all. This chapter reports on a deliberate effort to close that gap: instrumenting every turn with structured latency metrics (`stt_latency_ms`, `llm_ttft_ms`, `llm_total_ms`, `tts_first_chunk_ms`, `tts_total_ms`, `total_latency_ms`), persisting them per combination, and then running enough live test calls across every cell of the provider matrix to support a proper statistical analysis rather than an eyeballed comparison of averages.

The provider matrix under test:

- **STT** (speech-to-text): Sarvam, Groq (Whisper), Gemini
- **LLM**: Groq (Llama), Gemini, Anthropic (Claude)
- **TTS** (text-to-speech): ElevenLabs, Gemini, IndicF5 (self-hosted, fine-tuned), Sarvam

This yields 36 distinct layered-pipeline combinations. A 37th configuration, speech-to-speech (Gemini Live, a single end-to-end audio model with no discrete STT/LLM/TTS stages), is excluded from the factorial ANOVA in §3.1–3.2 since its latency semantics are not decomposable into the same three factors and are not directly comparable to the cascade architecture's per-stage timings. It is, however, included as a separate descriptive and comparative analysis in §3.3, since a direct latency comparison between the two architectures — factorial structure aside — is itself of practical interest.

## 2. Data and Methodology

### 2.1 Data Collection

Each of the 36 combinations was exercised via manual test calls through the platform's browser-based voice development console, with turn-level metrics persisted automatically to a `turn_metrics` table on every completed turn. Sample sizes per combination were deliberately evened up to a floor of approximately 10 turns each (ranging from 9 to 24 in the final dataset, owing to some combinations having accumulated more turns during earlier, less systematic testing) before this analysis was run, yielding **462 total turns** across all 36 combinations.

This is a field/convenience sample from real conversational test calls, not a fully randomized controlled experiment — test calls varied in conversational content, length, and time of day, and were not run in randomized order across combinations. The analysis below should accordingly be read as a robust *descriptive and comparative* account of what these combinations actually produced, rather than as a claim that every confound has been eliminated.

### 2.2 Metrics Analyzed

Two latency metrics were selected as the primary outcomes:

- **`tts_first_chunk_ms`** — time from the end of the caller's utterance to the first audio chunk of the agent's spoken reply. This is the metric most closely aligned with a caller's subjective sense of "how fast did it respond."
- **`total_latency_ms`** — time from the end of the caller's utterance to the completion of the entire turn (including full response synthesis). This captures whole-turn duration rather than perceived responsiveness alone.

### 2.3 Statistical Approach

Raw latency values are right-skewed (a small number of slow outlier turns pull the mean upward), which violates the normality-of-residuals assumption underlying ANOVA's F-test. Both metrics were therefore log-transformed (`log(1 + x)`) before analysis — a standard and appropriate transformation for latency/timing data, which tends to follow an approximately log-normal distribution. Reported group means below are back-transformed to milliseconds for interpretability.

Two complementary analyses were run for each metric:

1. **One-way ANOVA per factor** — testing each of STT, LLM, and TTS independently against the outcome, ignoring the other two factors. This is the simpler, more robust test and is appropriate given that some combinations still have modest sample sizes (9–24 turns).
2. **Three-way factorial ANOVA** (Type II sum of squares) — a single model including all three main effects and all three two-way interaction terms (STT×LLM, STT×TTS, LLM×TTS), which additionally tests whether the effect of one provider choice depends on which of the other two providers is also in use. Three-way interactions were not included, as the available sample size does not support reliable estimation of a 3×3×4 three-way interaction term.

Effect sizes are reported as η² (eta-squared: the proportion of total variance in the outcome attributable to a given factor), computed as each term's sum of squares divided by the total sum of squares across all terms plus the residual.

For the S2S comparison (§3.3), where there is only a single condition rather than multiple factor levels to vary, ANOVA does not apply. Instead, S2S is characterized descriptively (mean, median, standard deviation, quartiles) and compared against the best-performing layered combination using two independent two-sample tests: Welch's t-test (on log-transformed values, robust to unequal variances) and the Mann-Whitney U test (a non-parametric, distribution-free alternative), so that the conclusion does not rest on either test's assumptions alone.

## 3. Results

### 3.1 `tts_first_chunk_ms` — Perceived Responsiveness

**One-way ANOVA per factor:**

| Factor | F | p-value | Verdict |
|---|---:|---:|---|
| STT | 0.007 | 0.993 | Not significant |
| LLM | 0.577 | 0.562 | Not significant |
| TTS | 238.734 | <0.00001 | **Highly significant** |

Group means (back-transformed to milliseconds):

| STT | ms | | LLM | ms | | TTS | ms |
|---|---:|---|---|---:|---|---|---:|
| gemini | 2755 | | anthropic | 2657 | | elevenlabs | **1442** |
| groq | 2748 | | gemini | 2773 | | sarvam | 2279 |
| sarvam | 2769 | | groq | 2844 | | indicf5 | 3119 |
| | | | | | | gemini | **5629** |

**Three-way factorial ANOVA:**

| Term | Sum of Squares | df | F | p-value | η² |
|---|---:|---:|---:|---:|---:|
| STT | 0.011 | 2 | 0.044 | 0.957 | 0.0001 |
| LLM | 0.289 | 2 | 1.129 | 0.324 | 0.0018 |
| **TTS** | **96.365** | 3 | **251.053** | **<0.00001** | **0.6078** |
| STT × LLM | 0.590 | 4 | 1.152 | 0.332 | 0.0037 |
| STT × TTS | 3.002 | 6 | 3.910 | 0.0008 | 0.0189 |
| LLM × TTS | 2.249 | 6 | 2.930 | 0.0082 | 0.0142 |
| Residual | 56.041 | 438 | — | — | 0.3535 |

STT and LLM choice have no detectable main effect on time-to-first-response, individually or jointly with each other — consistent with the architecture of the cascade pipeline, in which TTS synthesis begins as soon as the LLM produces its first sentence, largely independent of which upstream STT or LLM produced that text. TTS provider choice alone accounts for **60.8%** of the explainable variance in this metric — by a wide margin the dominant factor. Both two-way interactions involving TTS (STT×TTS, LLM×TTS) are statistically significant, though their effect sizes are modest (1.9% and 1.4% of variance respectively) — the practical implication is that while TTS choice dominates overall, its exact latency shifts slightly depending on which STT or LLM is paired with it, rather than being perfectly additive.

### 3.2 `total_latency_ms` — Whole-Turn Duration

**One-way ANOVA per factor:**

| Factor | F | p-value | Verdict |
|---|---:|---:|---|
| STT | 12.645 | <0.00001 | **Significant** |
| LLM | 15.941 | <0.00001 | **Significant** |
| TTS | 260.202 | <0.00001 | **Highly significant** |

Group means (back-transformed to milliseconds):

| STT | ms | | LLM | ms | | TTS | ms |
|---|---:|---|---|---:|---|---|---:|
| groq | 7310 | | groq | **6559** | | elevenlabs | **3822** |
| sarvam | 7359 | | gemini | 9031 | | sarvam | 6231 |
| gemini | **10139** | | anthropic | 9490 | | indicf5 | 10119 |
| | | | | | | gemini | **17569** |

**Three-way factorial ANOVA:**

| Term | Sum of Squares | df | F | p-value | η² |
|---|---:|---:|---:|---:|---:|
| STT | 11.451 | 2 | 74.411 | <0.00001 | 0.0556 |
| LLM | 15.149 | 2 | 98.440 | <0.00001 | 0.0735 |
| **TTS** | **130.226** | 3 | **564.165** | **<0.00001** | **0.6319** |
| STT × LLM | 0.629 | 4 | 2.043 | 0.0874 | 0.0031 |
| STT × TTS | 5.989 | 6 | 12.972 | <0.00001 | 0.0291 |
| LLM × TTS | 8.950 | 6 | 19.387 | <0.00001 | 0.0434 |
| Residual | 33.701 | 438 | — | — | 0.1635 |

Unlike time-to-first-response, whole-turn duration is significantly affected by all three provider dimensions — expected, since total latency accumulates sequential STT, LLM, and TTS processing time rather than being dominated by whichever stage happens to gate the first audible response. TTS remains the largest single contributor (63.2% of variance), but STT (5.6%) and LLM (7.4%) are now both non-trivial. Both TTS-involving interaction terms are highly significant here (STT×TTS and LLM×TTS, both p<0.00001), with LLM×TTS the larger of the two (4.3% of variance) — meaning the total-latency cost of a slow TTS provider compounds differently depending on which LLM feeds it, rather than the two effects simply adding together.

### 3.3 Speech-to-Speech (S2S) Comparison

The platform also supports a speech-to-speech architecture (Gemini Live), in which a single end-to-end audio model both listens and speaks with no discrete STT, LLM, or TTS stages. Because this collapses the three-factor structure of the layered pipeline into a single condition, it cannot be entered into the ANOVA above as a factor level; it is instead reported descriptively and compared directly against the layered pipeline's fastest-performing combination.

An important measurement detail bears on interpretation here: S2S's `tts_first_chunk_ms` is anchored from the last detected caller speech this turn (a proxy for "the caller just finished talking"), a fix applied earlier in this project's benchmarking work after an initial version — anchored from turn-start, which included however long the caller had been speaking — produced numbers that did not match the subjectively fast feel of S2S calls. The figures below reflect the corrected, caller-speech-anchored measurement.

**Descriptive statistics (32 S2S turns):**

| Metric | n | mean | median | std | min | max | IQR (Q1–Q3) |
|---|--:|--:|--:|--:|--:|--:|---|
| `tts_first_chunk_ms` | 32 | **435ms** | 414ms | 302 | 0 | 1050 | 176–634 |
| `total_latency_ms` | 32 | 9988ms | 10100ms | 4878 | 476 | 25956 | 6883–12236 |

**Comparison against the best layered combination (ElevenLabs TTS, the fastest TTS provider identified in §3.1):**

| | n | mean `tts_first_chunk_ms` |
|---|--:|--:|
| S2S | 32 | **435ms** |
| ElevenLabs (best layered TTS, averaged across all STT/LLM pairings) | 101 | 1727ms |

- Welch's t-test (log-transformed): t = −6.174, **p < 0.00001**
- Mann-Whitney U (raw values, distribution-free): U = 16.0, **p < 0.00001**

Both tests — one parametric, one not — agree that S2S is statistically significantly faster than even the best layered combination, by a factor of roughly 4x (435ms vs. 1727ms mean). This is the first quantitative confirmation of an earlier subjective observation made during live testing: that S2S calls felt more responsive than layered calls, despite the layered pipeline's own turn-metrics initially appearing to tell a different story (an artifact of the pre-fix anchoring issue described above).

This advantage does not extend uniformly to whole-turn duration: S2S's mean `total_latency_ms` (9988ms) sits within the same range as the layered pipeline's slower TTS options (§3.2), rather than showing a comparable multiplicative advantage. This is expected given what `total_latency_ms` measures — the full turn including the agent's entire spoken response, not just the delay before it starts — so S2S's advantage is specific to *starting* its response quickly, not to how long the response takes to finish being spoken.

## 4. Discussion

Two distinct pictures emerge depending on which latency metric is treated as the priority.

**If perceived responsiveness is the priority** — the caller's subjective sense that the agent replied quickly — TTS provider selection is essentially the only lever that matters. STT and LLM choice can be selected on other grounds (cost, accuracy, language coverage) without meaningfully trading off response speed. Among the four TTS providers tested, ElevenLabs (1442ms average) is markedly faster than Sarvam (2279ms) and IndicF5 (3119ms), with Gemini TTS a clear outlier at 5629ms — roughly four times slower than ElevenLabs.

**If whole-turn duration is the priority** — for instance, in a context where the full response must complete within a hard time budget — all three provider choices matter, and the fastest combination is not simply "the fastest of each independently," given the significant STT×TTS and LLM×TTS interactions. The fastest observed configuration pairs Groq LLM (6559ms average) with Groq or Sarvam STT (both ~7300ms) and ElevenLabs TTS; Gemini STT is a consistent laggard (10139ms) regardless of what it's paired with.

The consistency of TTS as the dominant factor across both metrics — and specifically Gemini TTS as the slowest provider in both analyses — corroborates an earlier, less formal observation made during this project's broader benchmarking work: that LLM and TTS inference time, not STT, are the primary latency bottleneck in this pipeline. This analysis sharpens that finding considerably: it is specifically TTS, not LLM, that drives the majority of variance in what a caller actually experiences as responsiveness.

Set against this, the S2S comparison in §3.3 reframes the question somewhat: rather than asking "which layered combination responds fastest," the platform has a third option that outperforms every layered combination on responsiveness by a wide, statistically confirmed margin, precisely because it eliminates the separate TTS-synthesis step (and the STT/LLM steps preceding it) entirely. Where the layered analysis is about optimizing *within* a three-stage architecture, the S2S result is evidence that the architecture choice itself may matter more than any single provider swap within it — at least for responsiveness. Whole-turn duration, however, does not show the same advantage, so this reframing applies specifically to how quickly a call starts responding, not to how quickly a full response concludes.

## 5. Recommendations

- For any deployment where perceived call responsiveness is the primary concern, **ElevenLabs should be the default TTS choice** among the options tested, with Sarvam as the next-best alternative; **Gemini TTS should be avoided** unless its other qualities (e.g. voice quality, language support) are specifically required, given its roughly 4x latency penalty.
- STT and LLM provider selection can reasonably be driven by non-latency criteria (cost, transcription accuracy, language coverage, response quality) without materially affecting how fast a call feels to a caller.
- Where whole-turn duration matters (e.g. a strict time budget per turn), Groq is the fastest LLM option by a clear margin, and Gemini STT should be avoided in favor of Groq or Sarvam STT.
- Because interaction effects are statistically significant, a specific proposed combination should ideally be spot-checked directly rather than assumed purely from each factor's independent average — particularly for combinations pairing a slow TTS provider with a slow LLM, where the two effects appear to compound beyond simple addition.
- Where perceived responsiveness is the overriding priority and the S2S architecture's other trade-offs (cost, language/tool-calling support, control over conversational structure) are acceptable, **S2S should be preferred over any layered combination** — it is significantly faster to first response than even the best layered option, by roughly 4x. Where whole-turn duration or layered-pipeline-specific capabilities matter more, this advantage does not carry over, and the layered recommendations above still apply.

## 6. Limitations

- **Not a randomized controlled experiment.** Test calls were manual, conversational, and varied in length and content across combinations; they were not run in randomized order, and confounds such as time-of-day network conditions or subtly different conversational paths across test sessions cannot be fully ruled out.
- **Sample sizes, while balanced, remain modest** (9–24 turns per combination). This is adequate for detecting the large effects reported here (TTS's effect size, in particular, is very large by conventional standards) but should not be over-interpreted for the smaller, though still statistically significant, STT and LLM effects on total latency.
- **This analysis measures latency only.** It says nothing about response quality, transcription accuracy, pronunciation correctness, or conversational appropriateness — dimensions of provider comparison that are tracked separately in this project and are not synthesized into this statistical model.
- **Three-way interaction effects were not modeled**, given that a full 3×3×4 three-way interaction term would require substantially more data per cell than is currently available to estimate reliably.
- **The S2S sample is smaller than any individual layered combination** (32 turns, vs. 9–24 per layered cell) and, like the layered data, is a convenience sample from manual test calls rather than a randomized experiment. The magnitude of the S2S-vs-layered difference (roughly 4x, confirmed by two independent tests) is large enough that this sample size does not undermine the finding, but a larger S2S sample would still tighten the confidence interval around the exact effect size.

## 7. Conclusion

A two-way ANOVA across the platform's full supported STT/LLM/TTS provider matrix, run on 462 real test-call turns balanced across all 36 combinations, demonstrates that TTS provider selection is the dominant determinant of perceived voice-agent responsiveness (η² = 0.61), with STT and LLM choice contributing no measurable effect to that specific metric. For whole-turn duration, all three provider dimensions matter, with statistically significant interaction effects indicating that combination-level testing — not just independent per-provider comparison — is warranted when optimizing this pipeline's latency. A separate comparison against the platform's speech-to-speech architecture finds it to be significantly faster to first response than even the best layered combination (roughly 4x, confirmed by both a parametric and a non-parametric test), though this advantage is specific to response start time and does not extend to whole-turn duration.
