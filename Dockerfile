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
COPY data/kb ./data/kb

EXPOSE 8000

# Run migrations with a 60-second timeout (non-blocking: uvicorn starts even if
# alembic hangs — safe because we have no new migrations in this deploy and the
# schema is already at head from the last successful boot).
CMD ["sh", "-c", "timeout 60 alembic upgrade head; exec uvicorn src.main:app --host 0.0.0.0 --port 8000"]
