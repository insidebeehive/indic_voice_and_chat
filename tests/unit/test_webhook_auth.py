"""Unit tests for src/auth/webhook_auth.py.

Covers all four provider verification schemes (Twilio, Stringee, Exotel,
Chatwoot), the `signature_mode()` env-var toggle, and `constant_time_eq()`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time

import pytest
from twilio.request_validator import RequestValidator

from src.auth.webhook_auth import (
    WebhookAuthError,
    constant_time_eq,
    signature_mode,
    verify_chatwoot,
    verify_exotel_basic,
    verify_stringee,
    verify_twilio,
)

# --------------------------------------------------------------------------
# Twilio
# --------------------------------------------------------------------------


def test_twilio_valid_signature_real_sdk():
    token = "twilio_auth_token_123"
    url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"}
    sig = RequestValidator(token).compute_signature(url, params)

    assert verify_twilio([url], params, sig, token) is None


def test_twilio_tampered_param_rejected():
    token = "twilio_auth_token_123"
    url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"}
    sig = RequestValidator(token).compute_signature(url, params)

    tampered = dict(params)
    tampered["From"] = "+19998887777"

    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio([url], tampered, sig, token)
    assert exc.value.reason == "bad_signature"


def test_twilio_wrong_auth_token_rejected():
    url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123"}
    sig = RequestValidator("correct_token").compute_signature(url, params)

    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio([url], params, sig, "wrong_token")
    assert exc.value.reason == "bad_signature"


def test_twilio_missing_signature():
    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio(["https://example.com/x"], {}, None, "token")
    assert exc.value.reason == "missing_credentials"


def test_twilio_missing_auth_token():
    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio(["https://example.com/x"], {}, "some-sig", None)
    assert exc.value.reason == "missing_credentials"


def test_twilio_multiple_url_candidates_second_matches():
    token = "twilio_auth_token_123"
    wrong_url = "https://example.com/wrong-path"
    correct_url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123"}
    sig = RequestValidator(token).compute_signature(correct_url, params)

    assert verify_twilio([wrong_url, correct_url], params, sig, token) is None


# --------------------------------------------------------------------------
# Stringee
# --------------------------------------------------------------------------


def _stringee_sig(secret: str, data: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), data, hashlib.sha1)
    return base64.b64encode(mac.digest()).decode()


def test_stringee_post_valid():
    secret = "stringee_secret"
    body = b'{"a":1}'
    sig = _stringee_sig(secret, body)

    assert (
        verify_stringee(
            raw_body=body,
            url_path_and_query="/webhooks/stringee",
            signature=sig,
            signing_secret=secret,
        )
        is None
    )


def test_stringee_post_tampered_body_rejected():
    secret = "stringee_secret"
    body = b'{"a":1}'
    sig = _stringee_sig(secret, body)

    with pytest.raises(WebhookAuthError) as exc:
        verify_stringee(
            raw_body=b'{"a":2}',
            url_path_and_query="/webhooks/stringee",
            signature=sig,
            signing_secret=secret,
        )
    assert exc.value.reason == "bad_signature"


def test_stringee_get_valid():
    secret = "stringee_secret"
    path = "/webhooks/stringee?call_id=123"
    sig = _stringee_sig(secret, path.encode("utf-8"))

    assert (
        verify_stringee(
            raw_body=None,
            url_path_and_query=path,
            signature=sig,
            signing_secret=secret,
        )
        is None
    )


def test_stringee_get_different_path_rejected():
    secret = "stringee_secret"
    path = "/webhooks/stringee?call_id=123"
    sig = _stringee_sig(secret, path.encode("utf-8"))

    with pytest.raises(WebhookAuthError) as exc:
        verify_stringee(
            raw_body=None,
            url_path_and_query="/webhooks/stringee?call_id=999",
            signature=sig,
            signing_secret=secret,
        )
    assert exc.value.reason == "bad_signature"


def test_stringee_missing_signature():
    with pytest.raises(WebhookAuthError) as exc:
        verify_stringee(
            raw_body=b"body",
            url_path_and_query="/x",
            signature=None,
            signing_secret="secret",
        )
    assert exc.value.reason == "missing_credentials"


def test_stringee_missing_signing_secret():
    with pytest.raises(WebhookAuthError) as exc:
        verify_stringee(
            raw_body=b"body",
            url_path_and_query="/x",
            signature="somesig",
            signing_secret=None,
        )
    assert exc.value.reason == "missing_credentials"


# --------------------------------------------------------------------------
# Exotel
# --------------------------------------------------------------------------


def _basic_header(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def test_exotel_valid_credentials():
    header = _basic_header("exotel_user", "exotel_pass")
    assert (
        verify_exotel_basic(header, username="exotel_user", password="exotel_pass")
        is None
    )


def test_exotel_wrong_username_only():
    header = _basic_header("wrong_user", "exotel_pass")
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(header, username="exotel_user", password="exotel_pass")
    assert exc.value.reason == "bad_signature"


def test_exotel_wrong_password_only():
    header = _basic_header("exotel_user", "wrong_pass")
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(header, username="exotel_user", password="exotel_pass")
    assert exc.value.reason == "bad_signature"


def test_exotel_wrong_scheme():
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic("Bearer xyz", username="u", password="p")
    assert exc.value.reason == "malformed_credentials"


def test_exotel_bad_base64():
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic("Basic !!!not-valid-base64!!!", username="u", password="p")
    assert exc.value.reason == "malformed_credentials"


def test_exotel_no_colon_in_decoded():
    header = "Basic " + base64.b64encode(b"nodata").decode()
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(header, username="u", password="p")
    assert exc.value.reason == "malformed_credentials"


def test_exotel_missing_authorization_header():
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(None, username="u", password="p")
    assert exc.value.reason == "missing_credentials"


def test_exotel_missing_username_configured():
    header = _basic_header("u", "p")
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(header, username=None, password="p")
    assert exc.value.reason == "missing_credentials"


def test_exotel_missing_password_configured():
    header = _basic_header("u", "p")
    with pytest.raises(WebhookAuthError) as exc:
        verify_exotel_basic(header, username="u", password=None)
    assert exc.value.reason == "missing_credentials"


def test_exotel_lowercase_scheme_accepted():
    # RFC 7235 §2.1: the auth-scheme token is case-insensitive.
    header = "basic " + base64.b64encode(b"exotel_user:exotel_pass").decode()
    assert (
        verify_exotel_basic(header, username="exotel_user", password="exotel_pass")
        is None
    )


# --------------------------------------------------------------------------
# Chatwoot
# --------------------------------------------------------------------------


def _chatwoot_sig(secret: str, ts: int, raw_body: bytes) -> str:
    payload = f"{ts}.{raw_body.decode('utf-8')}".encode()
    return "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def test_chatwoot_valid_fresh_signature():
    secret = "chatwoot_secret"
    raw_body = b'{"event":"message_created"}'
    ts = int(time.time())
    sig = _chatwoot_sig(secret, ts, raw_body)

    assert (
        verify_chatwoot(
            raw_body=raw_body,
            secret=secret,
            signature_header=sig,
            timestamp_header=str(ts),
        )
        is None
    )


def test_chatwoot_stale_timestamp():
    secret = "chatwoot_secret"
    raw_body = b'{"event":"message_created"}'
    ts = int(time.time()) - 600
    sig = _chatwoot_sig(secret, ts, raw_body)

    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=raw_body,
            secret=secret,
            signature_header=sig,
            timestamp_header=str(ts),
            max_age_seconds=300,
        )
    assert exc.value.reason == "stale_timestamp"


def test_chatwoot_tampered_body():
    secret = "chatwoot_secret"
    raw_body = b'{"event":"message_created"}'
    ts = int(time.time())
    sig = _chatwoot_sig(secret, ts, raw_body)

    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b'{"event":"message_deleted"}',
            secret=secret,
            signature_header=sig,
            timestamp_header=str(ts),
        )
    assert exc.value.reason == "bad_signature"


def test_chatwoot_missing_secret():
    ts = int(time.time())
    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b"body",
            secret=None,
            signature_header="sha256=abc",
            timestamp_header=str(ts),
        )
    assert exc.value.reason == "no_secret_configured"


def test_chatwoot_non_integer_timestamp():
    secret = "chatwoot_secret"
    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b"body",
            secret=secret,
            signature_header="sha256=abc",
            timestamp_header="not-a-number",
        )
    assert exc.value.reason == "malformed_credentials"


def test_chatwoot_missing_signature_header():
    ts = int(time.time())
    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b"body",
            secret="secret",
            signature_header=None,
            timestamp_header=str(ts),
        )
    assert exc.value.reason == "missing_credentials"


def test_chatwoot_missing_timestamp_header():
    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b"body",
            secret="secret",
            signature_header="sha256=abc",
            timestamp_header=None,
        )
    assert exc.value.reason == "missing_credentials"


def test_chatwoot_non_utf8_body_raises_webhook_auth_error():
    # raw_body is attacker-controlled; a decode failure must not escape as a
    # raw UnicodeDecodeError.
    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=b"\xff\xfe",
            secret="s",
            signature_header="sha256=whatever",
            timestamp_header=str(int(time.time())),
        )
    assert exc.value.reason == "malformed_credentials"


def test_chatwoot_timestamp_exactly_at_max_age_boundary_passes():
    secret = "chatwoot_secret"
    raw_body = b'{"event":"message_created"}'
    max_age_seconds = 300
    ts = int(time.time()) - max_age_seconds
    sig = _chatwoot_sig(secret, ts, raw_body)

    assert (
        verify_chatwoot(
            raw_body=raw_body,
            secret=secret,
            signature_header=sig,
            timestamp_header=str(ts),
            max_age_seconds=max_age_seconds,
        )
        is None
    )


def test_chatwoot_future_dated_timestamp_rejected():
    secret = "chatwoot_secret"
    raw_body = b'{"event":"message_created"}'
    ts = int(time.time()) + 600
    sig = _chatwoot_sig(secret, ts, raw_body)

    with pytest.raises(WebhookAuthError) as exc:
        verify_chatwoot(
            raw_body=raw_body,
            secret=secret,
            signature_header=sig,
            timestamp_header=str(ts),
            max_age_seconds=300,
        )
    assert exc.value.reason == "stale_timestamp"


# --------------------------------------------------------------------------
# signature_mode()
# --------------------------------------------------------------------------


def test_signature_mode_unset_defaults_to_enforce(monkeypatch):
    monkeypatch.delenv("VOX_WEBHOOK_SIGNATURE_MODE", raising=False)
    assert signature_mode() == "enforce"


def test_signature_mode_log_only(monkeypatch):
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "log_only")
    assert signature_mode() == "log_only"


def test_signature_mode_case_insensitive(monkeypatch):
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "LOG_ONLY")
    assert signature_mode() == "log_only"


def test_signature_mode_invalid_value_falls_back_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "banana")
    with caplog.at_level(logging.WARNING):
        result = signature_mode()
    assert result == "enforce"
    assert any(
        record.levelno == logging.WARNING for record in caplog.records
    ), "expected a WARNING log for an unrecognized signature mode"


# --------------------------------------------------------------------------
# constant_time_eq()
# --------------------------------------------------------------------------


def test_constant_time_eq_equal():
    assert constant_time_eq("abc123", "abc123") is True


def test_constant_time_eq_unequal():
    assert constant_time_eq("abc123", "xyz789") is False


def test_constant_time_eq_none_a():
    assert constant_time_eq(None, "something") is False


def test_constant_time_eq_none_b():
    assert constant_time_eq("something", None) is False


def test_constant_time_eq_both_none():
    assert constant_time_eq(None, None) is False


def test_constant_time_eq_both_empty():
    # Empty strings are treated as "nothing to compare" (falsy), consistent
    # with treating an empty credential the same as a missing one — not as
    # two things that trivially compare equal.
    assert constant_time_eq("", "") is False


def test_constant_time_eq_non_ascii_equal():
    # Both inputs are UTF-8 encoded before comparing, so non-ASCII strings
    # must compare correctly rather than raising.
    assert constant_time_eq("café", "café") is True


# --------------------------------------------------------------------------
# Empty-string (falsy) credential inputs
# --------------------------------------------------------------------------


def test_twilio_empty_url_candidates_rejected():
    token = "twilio_auth_token_123"
    url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123"}
    sig = RequestValidator(token).compute_signature(url, params)

    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio([], params, sig, token)
    assert exc.value.reason == "bad_signature"


def test_twilio_empty_string_signature_treated_as_missing():
    with pytest.raises(WebhookAuthError) as exc:
        verify_twilio(["https://x"], {}, "", "tok")
    assert exc.value.reason == "missing_credentials"


def test_stringee_empty_string_signature_treated_as_missing():
    with pytest.raises(WebhookAuthError) as exc:
        verify_stringee(
            raw_body=b"x",
            url_path_and_query="/p",
            signature="",
            signing_secret="secret",
        )
    assert exc.value.reason == "missing_credentials"
