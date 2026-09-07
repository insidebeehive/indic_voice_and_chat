"""Observability push jobs (Phase 2: Prometheus/Grafana metrics).

Phase 1 (Loki log push) lives in ``src/utils/logging.py`` — unrelated to this
package. This package holds periodic aggregation-and-push jobs for metrics
(see ``turn_metrics_push.py``); later phases (health/readiness, trace-id) are
out of scope here.
"""

from __future__ import annotations
