"""Per-chat-turn latency/outcome metrics (Phase 2 of
docs/superpowers/plans/2026-09-08-chatbot-turn-metrics.md), giving ChatBot the
same durable aggregate-latency/failure insight VoiceBot already has via
``src/models/turn_metrics.py``.

Two sibling tables (deliberately NOT an extension of ``TurnMetric`` — see the
plan's §2 for the full argument; the short version is that voice's
provider-combo grouping in ``GET /benchmarks/turn-metrics/summary`` has no
``WHERE`` clause, so any chat row landing in ``turn_metrics`` would silently
corrupt that read path):

- ``ChatTurnMetric`` / ``chat_turn_metrics`` — one row per agent turn.
- ``ChatToolMetricRow`` / ``chat_tool_metrics`` — one row per tool call within
  a turn (child, FK CASCADE to the parent). Named ``...Row`` rather than
  ``ChatToolMetric`` to avoid colliding with the identically-named, unrelated
  ``@dataclass(frozen=True) ChatToolMetric`` in ``src/agents/chatbot.py``
  (Phase 1's in-process, unpersisted per-tool-call record). The two are NOT
  interchangeable: the dataclass lives entirely in the agent's memory for one
  turn and knows nothing about ``id``/``turn_id``/``tenant_id``/``created_at``;
  this module's ``ChatToolMetricRow`` is the DB row a dataclass instance gets
  turned into by ``record_chat_turn_metric`` below. Importers should never
  need both in the same file, but if one ever does, alias one of them (e.g.
  ``from src.agents.chatbot import ChatToolMetric as ChatToolMetricDC``) rather
  than shadowing.

Written best-effort from ``ChatBotAgent``'s injected ``record_metric``
callback (see ``src/agents/chatbot.py`` and ``src/bootstrap.py``'s
``make_chatbot_factory``) so a DB hiccup never affects a live chat turn — see
``record_chat_turn_metric``'s docstring, which mirrors
``src/models/turn_metrics.py::record_turn_metric``'s never-raises contract
exactly.

Integer columns are ``NOT NULL default 0`` and boolean columns are
``NOT NULL default false`` — same convention as ``TurnMetric`` uses, so
``AVG``/percentile queries over these columns never need a ``COALESCE``.

PII (non-negotiable — see the plan's §7): these tables store ONLY ids, tool
*names*, provider/model names, a small set of bounded enum strings (``path``,
``action``, ``kind``, ``outcome``), integers, and booleans. They must NEVER
store tool arguments, tool response bodies, message text, LLM completion
text, prompts, RAG chunk text, a customer/player id, or a CRM endpoint URL.
Critically, ``outcome`` is a bounded enum (``ok`` | ``timeout`` |
``transport_error`` | ``error`` | ``skipped_budget``) SPECIFICALLY so that a
free-text CRM error message — which can and does embed player data — has no
column to land in. Any future column proposal that could hold free text must
be rejected on that basis alone. Because there is nothing here that could
ever hold sensitive content, ``src/observability/trace_redaction.py`` is
deliberately NOT imported by this module — doing so would itself be a design
smell, signalling that content is entering the table. The correct property is
that there is nothing to redact.

``session_id`` is the live WebSocket capability for that chat session (same
status as in ``chat_messages``) — fine to store here, but it must NEVER be
returned by a read endpoint. There is no read endpoint in this module (Phase
2 is write-path only); when Phase 3 adds one, it must expose aggregates only,
never a raw ``session_id`` or ``trace_id``.

Known, accepted gap (Phase 2, not fixed here): a turn that raises or times
out writes no row. ``ChatBotAgent``'s turn coroutine can be cancelled by the
WS/HTTP layer's ``asyncio.wait_for`` timeout wrapper, and awaiting a DB insert
in a ``finally`` on that cancellation path is unreliable by construction (the
``await`` itself re-raises ``CancelledError``). The correct fix is a minimal
failure row written from the WS/HTTP layer's own timeout/error handler, which
knows tenant/session/trace-id/elapsed and isn't itself being cancelled —
that's Phase 3, deliberately out of scope here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base, get_sessionmaker
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# ChatToolMetricRow.tool_name's column width. Unlike every other string field
# here -- path/action/kind/outcome/provider/model are all bounded by
# construction (a fixed literal, an enum classifier, or a factory-closure
# value) -- tool_name is copied straight from the model's own tool_calls
# response (see src/agents/chatbot.py's ChatToolMetric dataclass), so a
# hallucinated or malformed tool name could in principle exceed this width.
# record_chat_turn_metric truncates to this length before insert so an
# over-long name costs a truncated label, not the loss of the entire row
# (Postgres raises StringDataRightTruncation on an overflow, which -- being
# swallowed by this module's never-raises contract -- would otherwise drop
# the parent row and all its children silently).
_TOOL_NAME_MAX_LEN = 100

# `action` values a chat_turn_metrics row carries when it represents a
# WS-layer turn failure (a turn that raised or timed out before the agent
# ever produced a ChatTurnResult -- see src/api/chat.py's
# _record_ws_turn_failure_metric) rather than a completed turn. Each value is
# "failed_" + one of src/api/chat.py::_classify_turn_error's fixed 4-value
# `reason` set -- never a free-text exception string (see this module's PII
# section above). Shared here, rather than duplicated in
# src/api/tenants.py's read endpoint and src/observability/chat_metrics_push.py
# (both of which need to tell a failure row apart from a completed turn so
# a turn that never ran does not silently drag down completed-turn averages),
# so all three call sites stay in sync if _classify_turn_error's reason set
# ever changes.
WS_TURN_FAILURE_ACTIONS = frozenset({
    "failed_llm_billing", "failed_llm_quota", "failed_timeout", "failed_internal",
})


class ChatTurnMetric(Base):
    __tablename__ = "chat_turn_metrics"
    __table_args__ = (
        # Both a standalone created_at index AND this composite are needed —
        # see alembic/versions/0019_turn_metrics_created_idx.py's own
        # docstring for why voice's table went a full migration without the
        # standalone index and paid for it with an ever-worsening full scan
        # in the periodic push loop. Don't repeat that here.
        Index("idx_chat_turn_metrics_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # CRM-level property (tool health), not turn-level — nullable since not
    # every tenant has a linked CRM. Available as tenant.settings.crm_id at
    # the write-path call site. Parent-only per the plan's §11.5 (a child-
    # table rollup can join to the parent; denormalizing there too is a
    # later call if that query becomes common).
    crm_id: Mapped[Optional[str]] = mapped_column(String(50))
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    # Nullable + indexed: not every turn necessarily has one resolved (e.g. a
    # very early failure before the trace context is bound), and this is the
    # single highest-leverage column here — it correlates a slow/bad row to
    # its Loki lines and (later) its Phoenix trace. See src/utils/trace_id.py.
    trace_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    path: Mapped[str] = mapped_column(String(20), nullable=False)  # "tools" | "single_shot"
    llm_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Prompt/completion/cached-prompt token counts for the turn. These exist
    # specifically to answer "is prompt caching actually happening" without
    # requiring log-scraping. cached_tokens should always be <= input_tokens
    # for a healthy row (not enforced at the DB level -- just the expectation).
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Budgeted CRM/deposit-verification tool time only — kb_search_ms below is
    # deliberately separate (it draws from its own independent timeout, not
    # the CRM tool-call budget); folding them would make tool_total_ms
    # incomparable to that budget. See src/agents/chatbot.py's own comments.
    tool_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_timeouts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kb_search_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kb_searches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieved_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rounds_exhausted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retry_fired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_directive_fired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_directive_escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    guard_hallucination_fired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    guard_no_grounding_fired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    guard_unverified_data_fired: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Length of the turn's FINAL customer-visible response_text, in chars and
    # whitespace-split words (see src/agents/chatbot.py's _measure_reply for
    # the exact definition, including the codepoint-vs-grapheme caveat).
    # Measured post-guard -- includes guard-substituted canned fallback text
    # when a guard fired, since that IS what the customer saw.
    #
    # NULLABLE, unlike every other integer column on this table (same
    # convention as ChatToolMetricRow.result_chars below). NULL means "no
    # measurement for this row": either it predates this migration (an older
    # agent process mid-rolling-deploy), or it's a WS-layer failure row (see
    # record_chat_turn_metric below and src/api/chat.py's
    # _record_ws_turn_failure_metric) that never produced a reply at all.
    # Backfilling either case to 0 would feed a false observation into any
    # AVG/percentile query over this column.
    #
    # UNLIKE result_chars, 0 IS a legitimate, representable value here: it
    # means "measured, reply genuinely empty" -- an alarming observation
    # (the customer got nothing back), not a missing one -- and it must stay
    # distinguishable from NULL ("we don't know"). AVG and percentile_cont
    # both skip NULLs, so a genuine 0 correctly drags those statistics down
    # while a NULL correctly doesn't.
    reply_chars: Mapped[Optional[int]] = mapped_column(Integer)
    reply_words: Mapped[Optional[int]] = mapped_column(Integer)
    # Indexed (both standalone, above via index=True on the column, and via
    # the composite in __table_args__) — see the class-level comment on why
    # both are required from day one.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), index=True,
    )


class ChatToolMetricRow(Base):
    """DB row for one tool call within a turn. See the module docstring for
    why this is named ``...Row`` rather than ``ChatToolMetric`` (that name is
    already taken by the unrelated in-process dataclass in
    ``src/agents/chatbot.py``)."""

    __tablename__ = "chat_tool_metrics"
    __table_args__ = (
        Index("idx_chat_tool_metrics_name_created", "tool_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_turn_metrics.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Denormalized from the parent (same convention TurnMetric uses for its
    # own tenant_id) so a tenant-scoped tool-latency query needs no join.
    tenant_id: Mapped[str] = mapped_column(String(50), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(_TOOL_NAME_MAX_LEN), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)  # "crm"|"kb"|"local"|"deposit_verification"
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # "ok" | "timeout" | "transport_error" | "error" | "skipped_budget" —
    # bounded enum, deliberately never a free-text error string. See the
    # module docstring's PII section.
    outcome: Mapped[str] = mapped_column(String(30), nullable=False)
    budget_slice_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    round_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Characters of this tool result's JSON as sent to the model (see
    # src/agents/chatbot.py's ChatToolMetric.result_chars). A LENGTH, never
    # content -- there is still nothing here to redact.
    #
    # NULLABLE, unlike every other integer column on these tables. Deliberate:
    # NULL means "this row predates the measurement", and rows written before
    # this migration would otherwise backfill to a literal 0 that AVG/
    # percentile would average in as a real observation -- biasing downward
    # exactly the statistic this column exists to produce. 0 is never a
    # legitimate value either: json.dumps of a result dict is at least "{}"
    # (2 chars). AVG and percentile_cont both skip NULLs, so no COALESCE is
    # needed for this one.
    result_chars: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(),
    )


async def record_chat_turn_metric(
    *,
    tenant_id: str,
    crm_id: Optional[str],
    session_id: str,
    trace_id: Optional[str],
    path: str,
    llm_provider: str,
    llm_model: str,
    action: str,
    metrics: dict[str, Any],
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> None:
    """Insert one ``chat_turn_metrics`` row and its ``chat_tool_metrics``
    children. Best-effort: never raises — a DB outage must degrade to
    no-persistence, not break a live chat turn (see ``ChatBotAgent``'s
    ``record_metric`` callback, the only caller, and
    ``src/models/turn_metrics.py::record_turn_metric``, whose contract this
    mirrors exactly).

    ``metrics`` holds the parent row's aggregate int/bool fields (as a plain
    dict, keyed by column name — read with ``.get(..., 0/False)`` so a
    partial dict never raises a KeyError, same convention as
    ``record_turn_metric``). ``tools`` holds zero or more per-tool-call dicts
    (``tool_name``/``kind``/``latency_ms``/``outcome``/``budget_slice_ms``/
    ``round_index``/``result_chars``), inserted as ``chat_tool_metrics`` rows
    once the parent's id is known.
    """
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            row = ChatTurnMetric(
                tenant_id=tenant_id,
                crm_id=crm_id,
                session_id=session_id,
                trace_id=trace_id,
                path=path,
                llm_provider=llm_provider,
                llm_model=llm_model,
                action=action,
                total_ms=metrics.get("total_ms", 0),
                llm_total_ms=metrics.get("llm_total_ms", 0),
                llm_calls=metrics.get("llm_calls", 0),
                input_tokens=metrics.get("input_tokens", 0),
                output_tokens=metrics.get("output_tokens", 0),
                cached_tokens=metrics.get("cached_tokens", 0),
                tool_total_ms=metrics.get("tool_total_ms", 0),
                tool_calls=metrics.get("tool_calls", 0),
                tool_failures=metrics.get("tool_failures", 0),
                tool_timeouts=metrics.get("tool_timeouts", 0),
                tool_calls_skipped=metrics.get("tool_calls_skipped", 0),
                kb_search_ms=metrics.get("kb_search_ms", 0),
                kb_searches=metrics.get("kb_searches", 0),
                retrieved_chunks=metrics.get("retrieved_chunks", 0),
                rounds=metrics.get("rounds", 0),
                rounds_exhausted=metrics.get("rounds_exhausted", False),
                retry_fired=metrics.get("retry_fired", False),
                failure_directive_fired=metrics.get("failure_directive_fired", False),
                failure_directive_escalated=metrics.get("failure_directive_escalated", False),
                guard_hallucination_fired=metrics.get("guard_hallucination_fired", False),
                guard_no_grounding_fired=metrics.get("guard_no_grounding_fired", False),
                guard_unverified_data_fired=metrics.get("guard_unverified_data_fired", False),
                escalated=metrics.get("escalated", False),
                # DELIBERATE DEPARTURE from every other field on this parent
                # row above, which all use `.get(key, 0/False)`: no default
                # here means a payload that omits these keys stores NULL, not
                # 0. That's exactly what makes a WS-layer failure row
                # (_record_ws_turn_failure_metric sends only total_ms/action)
                # come out NULL "for free" -- matching the column's own
                # NULL-vs-0 contract documented on ChatTurnMetric above. The
                # only other no-default precedent in this module is on the
                # CHILD row (ChatToolMetricRow.result_chars below), not here
                # -- don't "fix" this back to a uniform `.get(key, 0)`, or
                # every future AVG/percentile over reply length silently
                # starts treating unmeasured rows as empty replies.
                reply_chars=metrics.get("reply_chars"),
                reply_words=metrics.get("reply_words"),
            )
            db.add(row)
            # Flush (not commit) to allocate row.id without ending the
            # transaction, so the parent + all children commit atomically.
            await db.flush()
            child_rows = [
                ChatToolMetricRow(
                    turn_id=row.id,
                    tenant_id=tenant_id,
                    # Defensively truncated -- see _TOOL_NAME_MAX_LEN's
                    # comment: this is the one field here not already bounded
                    # by construction, since it comes straight from the
                    # model's own tool-call response.
                    tool_name=t["tool_name"][:_TOOL_NAME_MAX_LEN],
                    kind=t["kind"],
                    latency_ms=t.get("latency_ms", 0),
                    outcome=t["outcome"],
                    budget_slice_ms=t.get("budget_slice_ms", 0),
                    round_index=t.get("round_index", 0),
                    # .get with no default -> None (not 0) when the payload
                    # predates this field, e.g. an older agent process during
                    # a rolling deploy. Same reason the column is nullable:
                    # a missing measurement must not look like a 0-char result.
                    result_chars=t.get("result_chars"),
                )
                for t in tools
            ]
            if child_rows:
                db.add_all(child_rows)
            await db.commit()
            # The only place this write's success is ever recorded -- a
            # metric that fails to persist is invisible by definition, and so
            # (less obviously) is one that succeeds silently: without this,
            # "is this turn's metric even being written" needs a database
            # query to answer.
            debug_event(
                log, "metrics chat_turn_metric_write response",
                turn_id=row.id, tenant_id=tenant_id, session_id=session_id,
                trace_id=trace_id, path=path, action=action,
                tool_row_count=len(child_rows),
            )
    except Exception:  # noqa: BLE001 - must never break a live chat turn
        log.warning(
            "record_chat_turn_metric failed; continuing without persistence", exc_info=True,
        )
        # The WARNING above carries no correlation id, so "which turn's
        # metric just silently vanished" otherwise needs a second incident to
        # notice and a source read to explain -- see docs/debug-logging.md's
        # own motivating example for the skip category.
        debug_event(
            log, "metrics chat_turn_metric_write failed",
            tenant_id=tenant_id, session_id=session_id, trace_id=trace_id,
            path=path, action=action,
        )
