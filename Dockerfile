# Multi-stage: build deps in a throwaway builder, copy only the venv to a clean
# runtime so build tools never ship. SileroVAD runs on onnxruntime + the bundled
# ONNX model only — src/pipeline/vad.py never imports torch — so silero-vad is
# installed with --no-deps to keep torch/torchaudio (and ~4-5GB of Linux CUDA
# libraries) out of the image entirely.

# ---- builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install onnxruntime \
    && pip install --no-deps silero-vad

# ---- runtime ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY static ./static
COPY alembic.ini ./
COPY alembic ./alembic

EXPOSE 8000

# Apply DB migrations before serving, then exec the app (so uvicorn is PID 1 and
# receives signals). Fail-fast: a failed migration must not start a stale app.
# `alembic upgrade head` is idempotent; env.py skips CREATE SCHEMA when the schema
# already exists, so a least-privilege DB role can run it.
# NOTE: for multi-replica deploys, run migrations as a single pre-deploy job
# instead (Alembic has no cross-process lock).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn src.main:app --host 0.0.0.0 --port 8000"]
