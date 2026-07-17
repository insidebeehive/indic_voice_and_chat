# Self-hosted LLM (vLLM) on the IndicF5 RunPod pod

> Status: experiment. Goal: measure whether a self-hosted model can handle the
> live chat/voice turns (quality + latency vs Gemini), and add a self-hosted
> data point to the LLM experiments doc. No provider quota — the ceiling is
> the GPU.

## Pod

RTX PRO 4500 (32GB VRAM, 62GB RAM, 32 vCPU). IndicF5 TTS uses ~3-4GB, leaving
~28GB. The launch command below caps vLLM at ~55% of VRAM (~17GB) so TTS —
which is on the voice critical path — always has headroom. If TTS latency
spikes during LLM load, lower `--gpu-memory-utilization` first.

## Launch command (run on the pod)

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

- **Model**: Qwen2.5-14B-Instruct-AWQ (~10GB quantized) — best JSON +
  function-calling reliability at this size. Indic-focused alternative to
  benchmark later: `sarvamai/sarvam-m` (AWQ) — swap the model id and check
  which `--tool-call-parser` its card recommends.
- `--max-model-len 8192` covers our chat prompts (~4K tokens with RAG) and
  caps KV-cache growth.
- `--enable-auto-tool-choice --tool-call-parser hermes` is required for the
  chat agent's CRM tools; `hermes` is the parser for Qwen 2.5 models.
- Expose port **8001** in the RunPod config (proxy URL becomes
  `https://<pod-id>-8001.proxy.runpod.net`).

Smoke check from anywhere:

```bash
curl https://<pod-id>-8001.proxy.runpod.net/v1/models
```

## Platform config

```bash
# .env / platform env — NOTE the /v1 suffix
VLLM_BASE_URL=https://<pod-id>-8001.proxy.runpod.net/v1
```

Provider key is `vllm` (adapter: `src/providers/llm/openai_compat.py`,
default model `Qwen/Qwen2.5-14B-Instruct-AWQ`).

## Running the experiments

- **Voice (per-tenant)**: set the dev tenant's `pipeline_config.llm` to
  `{"provider": "vllm"}` — the voice cascade honors per-tenant LLM config.
- **Chat (platform-level)**: chat always uses the platform LLM, so point
  `config/default.yaml` `pipeline.llm.provider: vllm` in a LOCAL run and use
  `scripts/chat_load_test.py` against it. Don't change the staging default
  until quality is proven.
- Record latency/quality numbers in `docs/latency-llm-stt-experiments.md`.

## Known limits

- Text-only: the adapter drops image parts (chat images stay on Gemini).
- No `transcribe_audio` — call-outcome transcription stays on Gemini.
- One pod = TTS + LLM share fate: a pod restart takes down both. Fine for an
  experiment; a production adoption gets its own inference pod.
