"""Provider → telephony answer-webhook route path (under ``/api/v1/telephony/``).

Shared by the dev-console ``place-call`` and the campaign ``call_lead`` paths so
the OUTBOUND answer URL is slug-scoped consistently: ``{base}/{path}/{slug}``.
For a call WE place the tenant is known, so the answer webhook resolves by slug
(``tenant_from_slug``) instead of reverse-resolving our own caller-ID — which
would require the number to be registered in ``tenant_phone_numbers``.
"""

from __future__ import annotations

ANSWER_PATHS: dict[str, str] = {
    "twilio": "twilio/voice",
    "exotel": "exotel/voice",
    "stringee": "stringee/answer",
}
"""Providers that have a slug-scoped answer route. A provider absent here gets a
bare base URL (unchanged legacy behaviour)."""
