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

# Migrations must reach head before the app serves. This used to be `;` rather
# than `&&`, so a failed migration started uvicorn anyway against an unmigrated
# schema — and that hid the same bug twice: a revision id longer than
# alembic_version.version_num's VARCHAR(32) (0019, then 0025), each time
# leaving the app running for hours against columns that did not exist, with
# nothing failing except whatever quietly swallowed the resulting errors.
#
# Retried rather than failed on the first attempt, because the `;` was not
# arbitrary: bb6c6a4 introduced it to break a real Northflank crash loop where
# alembic blocked on a DB connection or lock still held by the OUTGOING
# container during a rolling restart. That condition is transient and clears in
# seconds, so three attempts absorb it. A genuine migration error fails all
# three in well under a minute and the container exits non-zero, which is what
# stops the rollout.
#
# 120s per attempt rather than 60: a premature kill now fails the deploy, so
# the cap has to be generous enough not to guillotine a legitimately slow
# migration.
#
# Unset DATABASE_URL skips migrations rather than failing. That is not a
# loophole in the rule above: alembic/env.py falls back to config/default.yaml's
# localhost URL when the variable is absent, so enforcing here would kill any
# deployment that legitimately runs without a database addon (the no-addon
# smoke-test stage in docs/deploy/northflank.md). A URL that IS set and cannot
# be reached still fails all three attempts and stops the rollout -- the
# distinction is "no database configured" versus "the configured database is
# unreachable", and only the second is a deployment error.
CMD ["sh", "-c", "if [ -z \"$DATABASE_URL\" ]; then echo \"DATABASE_URL unset - skipping migrations (no database configured)\"; else ok=0; for i in 1 2 3; do timeout 120 alembic upgrade head && { ok=1; break; }; echo \"alembic upgrade head failed (attempt $i/3)\"; sleep 10; done; [ \"$ok\" = 1 ] || { echo 'FATAL: migrations did not reach head; refusing to start on an unmigrated schema'; exit 1; }; fi; exec uvicorn src.main:app --host 0.0.0.0 --port 8000"]
