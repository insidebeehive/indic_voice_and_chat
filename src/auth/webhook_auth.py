"""Inbound webhook authentication for third-party providers.

Standalone verification helpers for the webhook signature/credential schemes
used by our telephony and CRM integrations: Twilio (HMAC-SHA1 via the Twilio
SDK's ``RequestValidator``), Stringee (HMAC-SHA1, base64), Exotel (HTTP Basic
auth), and Chatwoot (HMAC-SHA256 with a timestamp window against replay).

Each ``verify_*`` function raises :class:`WebhookAuthError` on failure and
returns ``None`` on success. None of these functions are wired into a live
route yet — that happens in a later PR; this module is isolated infra.

See :func:`signature_mode` for the enforce-vs-log-only toggle a caller can
use once these are wired in.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import time

from twilio.request_validator import RequestValidator

log = logging.getLogger(__name__)

_VALID_SIGNATURE_MODES = {"enforce", "log_only"}
_DEFAULT_SIGNATURE_MODE = "enforce"


def signature_mode() -> str:
    """Whether webhook signature failures should be enforced (reject) or only
    logged.

    Reads ``VOX_WEBHOOK_SIGNATURE_MODE`` from the environment fresh on every
    call (no caching) so tests can monkeypatch it. Value is stripped and
    lowercased before validation. Must be one of ``"enforce"`` or
    ``"log_only"``; an unset, empty, or unrecognized value falls back to the
    fail-safe default ``"enforce"`` — an unrecognized value also emits a
    WARNING log so misconfiguration doesn't silently pass as either mode.
    """
    raw = os.environ.get("VOX_WEBHOOK_SIGNATURE_MODE", _DEFAULT_SIGNATURE_MODE)
    value = raw.strip().lower()
    if value not in _VALID_SIGNATURE_MODES:
        log.warning(
            "Unrecognized VOX_WEBHOOK_SIGNATURE_MODE=%r, falling back to %r",
            raw,
            _DEFAULT_SIGNATURE_MODE,
        )
        return _DEFAULT_SIGNATURE_MODE
    return value


class WebhookAuthError(Exception):
    """Raised by any ``verify_*`` function on failure.

    Carries a ``reason`` attribute — a short machine-readable string (e.g.
    ``"bad_signature"``, ``"missing_credentials"``, ``"no_secret_configured"``,
    ``"malformed_credentials"``, ``"stale_timestamp"``) — for logging by the
    caller. This list is illustrative, not exhaustive.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason)


def constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string comparison that never raises.

    Wraps ``hmac.compare_digest``. Both inputs are encoded to UTF-8 bytes
    before comparing (avoids ``compare_digest``'s ASCII-only restriction on
    ``str`` arguments). Returns ``False`` if either value is ``None`` or
    empty (treated as "nothing to compare" rather than "compare equal"), or
    on any error (e.g. a non-string input) — this function must never raise.
    """
    if not a or not b:
        return False
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except (TypeError, AttributeError, UnicodeError):
        return False


def verify_twilio(
    url_candidates: list[str],
    params: dict[str, str],
    signature: str | None,
    auth_token: str | None,
) -> None:
    """Verify an inbound Twilio webhook's ``X-Twilio-Signature`` header.

    Raises :class:`WebhookAuthError` unless ``signature`` validates against
    at least one of ``url_candidates`` (tried in order — a proxy may present
    a different scheme/host than what Twilio originally signed) using
    ``twilio.request_validator.RequestValidator``. Never hand-rolls the HMAC.
    """
    if not signature or not auth_token:
        raise WebhookAuthError("missing_credentials")

    validator = RequestValidator(auth_token)
    for url in url_candidates:
        try:
            if validator.validate(url, params, signature):
                return
        except Exception:  # noqa: BLE001 - one bad candidate must not abort the loop
            continue
    raise WebhookAuthError("bad_signature")


def verify_stringee(
    *,
    raw_body: bytes | None,
    url_path_and_query: str,
    signature: str | None,
    signing_secret: str | None,
) -> None:
    """Verify an inbound Stringee webhook signature.

    Raises :class:`WebhookAuthError` unless ``base64(HMAC-SHA1(signing_secret,
    data))`` constant-time-equals ``signature``, where ``data`` is
    ``raw_body`` for a POST (``raw_body is not None``) or
    ``url_path_and_query`` (UTF-8 encoded) for a GET (``raw_body is None``).

    Deliberately a simpler scheme than Chatwoot's: SHA1 (not SHA256), plain
    base64 (not hex), and no ``"sha256="``-style prefix.
    """
    if not signature or not signing_secret:
        raise WebhookAuthError("missing_credentials")

    data = raw_body if raw_body is not None else url_path_and_query.encode("utf-8")
    mac = hmac.new(signing_secret.encode("utf-8"), data, hashlib.sha1)
    expected = base64.b64encode(mac.digest()).decode("ascii")

    if not constant_time_eq(expected, signature):
        raise WebhookAuthError("bad_signature")


def verify_exotel_basic(
    authorization_header: str | None,
    *,
    username: str | None,
    password: str | None,
) -> None:
    """Verify an inbound Exotel webhook's HTTP Basic ``Authorization`` header.

    Raises :class:`WebhookAuthError` unless the header carries ``Basic
    <base64>`` decoding to ``username:password`` matching the configured
    ``username``/``password`` (split on the first ``:`` only). Both halves
    are compared with :func:`constant_time_eq` unconditionally (no
    short-circuit) to avoid a timing side-channel revealing which half was
    wrong.
    """
    if not username or not password:
        raise WebhookAuthError("missing_credentials")

    if not authorization_header:
        raise WebhookAuthError("missing_credentials")

    scheme, _, encoded = authorization_header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        raise WebhookAuthError("malformed_credentials")

    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise WebhookAuthError("malformed_credentials") from None

    got_user, sep, got_pass = decoded.partition(":")
    if not sep:
        raise WebhookAuthError("malformed_credentials")

    user_ok = constant_time_eq(got_user, username)
    pass_ok = constant_time_eq(got_pass, password)
    if not (user_ok and pass_ok):
        raise WebhookAuthError("bad_signature")


def verify_chatwoot(
    *,
    raw_body: bytes,
    secret: str | None,
    signature_header: str | None,
    timestamp_header: str | None,
    max_age_seconds: int = 300,
) -> None:
    """Verify an inbound Chatwoot webhook signature + timestamp freshness.

    Raises :class:`WebhookAuthError` unless ``signature_header == 'sha256=' +
    hex(HMAC-SHA256(secret, f'{timestamp_header}.{raw_body.decode()}'))``
    (constant-time compared) AND ``timestamp_header`` parses as an int within
    ``max_age_seconds`` of ``time.time()`` (rejects both stale and
    too-far-future timestamps — clock skew cuts both ways).

    Chatwoot HMAC is optional per-tenant, so a missing ``secret`` is its own
    distinct reason (``no_secret_configured``) rather than treated as an
    automatic pass — the caller decides what to do with that.
    """
    if not secret:
        raise WebhookAuthError("no_secret_configured")

    if not signature_header or not timestamp_header:
        raise WebhookAuthError("missing_credentials")

    try:
        ts = int(timestamp_header)
    except (TypeError, ValueError):
        raise WebhookAuthError("malformed_credentials") from None

    now = int(time.time())
    if abs(now - ts) > max_age_seconds:
        raise WebhookAuthError("stale_timestamp")

    try:
        payload = f"{timestamp_header}.{raw_body.decode('utf-8')}".encode()
        expected = (
            "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        )
    except UnicodeDecodeError:
        raise WebhookAuthError("malformed_credentials") from None

    if not constant_time_eq(expected, signature_header):
        raise WebhookAuthError("bad_signature")
