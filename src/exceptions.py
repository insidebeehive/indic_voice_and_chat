"""Shared exception types with no ``src.*`` dependencies of their own.

This module exists solely to break an import cycle between ``src.bootstrap``
and ``src.api`` (``src.bootstrap`` imports telephony bridge modules from
``src.api``, which import back from ``src.bootstrap`` for exception types
raised by bootstrap-built factories). Anything defined here must stay free of
imports from the rest of ``src`` — that's the whole point. Do NOT move these
back into ``src.bootstrap`` or otherwise reintroduce a dependency edge here;
that's exactly the cycle this module was carved out to avoid.
"""

from __future__ import annotations


class LiveKitModeNotSupported(Exception):
    """Raised by ``make_livekit_bridge_factory``'s factory when the tenant isn't
    in s2s pipeline mode. LiveKit room-join is s2s-only — there is no cascade
    (STT->LLM->TTS) LiveKit path, and no per-call mode override for it."""
