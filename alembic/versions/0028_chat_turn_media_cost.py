"""alembic/versions/0028_chat_turn_media_cost.py

Adds six platform-paid voice-note media cost columns to ``chat_turn_metrics``:
``tts_provider``/``tts_model``/``tts_audio_ms``/``tts_cost`` for a voice-note
reply's TTS synthesis, and ``stt_audio_ms``/``stt_cost`` for an inbound
voice-note's STT transcription (``stt_provider``/``stt_model`` are not added
here — the turn's own ``llm_provider``/``llm_model`` columns already carry
that identity for the STT path, since transcription and the turn's LLM are
typically the same provider; see ``src/api/chat.py``'s
``_record_turn_media_metrics``). See ``src/api/chat_cost.py`` for how these
costs are computed and ``src/models/chat_turn_metrics.py``'s ``ChatTurnMetric``
for the same NULL/0 semantics restated next to the columns.

NULLABLE with no server default and no backfill, same convention as
migration 0026's ``reply_chars``/``reply_words``:

- NULL on ``tts_provider``/``tts_model``/``tts_audio_ms``/``tts_cost`` means
  no TTS ran on this turn (text-only reply, TTS disabled/no provider
  configured) — or the row predates this migration.
- ``tts_audio_ms = 0`` together with ``tts_cost = 0`` is a distinct,
  deliberate state: the TTS provider WAS called but synthesis did not
  complete (timeout or raised) before a usable audio result came back, so
  the billed amount is unknown. This is the "flag" this feature needs for
  that case — no separate boolean column, since ``tts_provider``/
  ``tts_model`` are still populated (the provider was in fact invoked) while
  ``tts_audio_ms``/``tts_cost`` are 0 rather than NULL, and that combination
  is exactly what distinguishes it from "TTS never ran" (all four NULL).
- NULL on ``stt_audio_ms`` together with a non-NULL ``stt_cost`` means
  token-priced transcription (the provider reported usage, so cost was
  computed without needing an audio duration) whose audio duration is
  unknown/not measured — see ``_inbound_audio_duration_ms`` in
  ``src/api/chat.py``, which only decodes WAV.

No PII: only provider/model names (short catalog strings, same as the
existing ``llm_provider``/``llm_model`` columns) and numbers.

Revision: 0028_chat_turn_media_cost
Down: 0027_webhook_outbox
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0028_chat_turn_media_cost"
down_revision = "0027_webhook_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_turn_metrics",
        sa.Column("tts_provider", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("tts_model", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("tts_audio_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("tts_cost", sa.Float(), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("stt_audio_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("stt_cost", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_turn_metrics", "stt_cost")
    op.drop_column("chat_turn_metrics", "stt_audio_ms")
    op.drop_column("chat_turn_metrics", "tts_cost")
    op.drop_column("chat_turn_metrics", "tts_audio_ms")
    op.drop_column("chat_turn_metrics", "tts_model")
    op.drop_column("chat_turn_metrics", "tts_provider")
