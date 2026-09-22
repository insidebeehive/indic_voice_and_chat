"""No credential-resolution path may log the credential.

`docs/debug-logging.md` allows full values at DEBUG with exactly one absolute
exception: credentials appear at no level. `src/auth/` is where they are
decrypted and handed out, so this is where that rule is easiest to break and
hardest to notice -- a leak here is invisible until DEBUG is switched on during
an incident, which is when the most data is being shipped to Loki.

A sibling test already pins this for `TenantSettings.secret()`
(tests/unit/test_tenant_config.py::test_secret_resolution_debug_log_never_carries_the_value).
It does NOT cover the paths here, and that gap was not theoretical: while the
`src/auth/` instrumentation was being written, adding `value=value` to
`secrets.decrypt()`'s event leaked a full plaintext tenant credential into the
JSON log line, and nothing in the suite failed. `tests/unit/test_secrets.py`
uses `caplog` nowhere at all.

Each test below fails if the corresponding `debug_event` is given the raw
value, and also asserts the event stays USEFUL -- a fingerprint and a length,
so an operator can confirm which value resolved without being able to
reconstruct it. A test that only checked "the secret is absent" would pass
against an event that had been gutted to carry nothing.
"""

from __future__ import annotations

import logging

import pytest

from src.auth import secrets as auth_secrets
from src.auth.context import TenantContext
from src.config_tenant import TenantSettings

CANARY = "AC-canary-leak-9f3b2c1d8e7a6"


def _logged_text(caplog: pytest.LogCaptureFixture) -> str:
    """Message text AND every structured field, since `debug_event` puts the
    values in `extra` -- a leak hides in the record attributes, not the
    formatted message."""
    return "\n".join(r.getMessage() + " " + repr(r.__dict__) for r in caplog.records)


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VOX_SECRET_KEY", auth_secrets.generate_key())
    auth_secrets.reset_cache_for_tests()
    yield
    auth_secrets.reset_cache_for_tests()


def test_encrypt_does_not_log_the_plaintext(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    token = auth_secrets.encrypt(CANARY)
    assert token and token != CANARY
    assert CANARY not in _logged_text(caplog), "encrypt() logged the plaintext secret"


def test_decrypt_does_not_log_the_plaintext(caplog: pytest.LogCaptureFixture) -> None:
    """The demonstrated leak: decrypt() holds the plaintext by definition."""
    token = auth_secrets.encrypt(CANARY)
    caplog.clear()
    caplog.set_level(logging.DEBUG)

    assert auth_secrets.decrypt(token) == CANARY

    logged = _logged_text(caplog)
    assert CANARY not in logged, "decrypt() logged the plaintext secret"
    # ...and the event is still worth having.
    assert "value_len" in logged, "decrypt()'s event carries no length to identify the value by"


def test_encrypted_token_itself_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The ciphertext is not plaintext, but it is still the stored credential:
    anything holding it plus the Fernet key has the secret, and both travel to
    the same Loki instance. Lengths only."""
    caplog.set_level(logging.DEBUG)
    token = auth_secrets.encrypt(CANARY)
    caplog.clear()
    auth_secrets.decrypt(token)
    assert token not in _logged_text(caplog), "the encrypted token was logged verbatim"


def test_fernet_key_is_never_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """`secrets fernet_key loaded` fires once per process and is the single
    place the master key is in hand. Logging it would compromise every stored
    secret at once, not one."""
    key = auth_secrets.generate_key()
    monkeypatch.setenv("VOX_SECRET_KEY", key)
    auth_secrets.reset_cache_for_tests()
    caplog.set_level(logging.DEBUG)

    auth_secrets.encrypt("x")

    assert key not in _logged_text(caplog), "the Fernet master key reached a log line"


def test_tenant_context_secret_does_not_log_the_value(caplog: pytest.LogCaptureFixture) -> None:
    """`TenantContext.secret()` serves an already-resolved per-tenant secret,
    a different path from TenantSettings.secret() and not covered by its test."""
    caplog.set_level(logging.DEBUG)
    ctx = TenantContext(
        settings=TenantSettings(id="t_canary", slug="canary", name="Canary"),
        secrets_resolved={"TWILIO_AUTH_TOKEN": CANARY},
    )

    assert ctx.secret("TWILIO_AUTH_TOKEN") == CANARY

    logged = _logged_text(caplog)
    assert CANARY not in logged, "TenantContext.secret() logged the value"
    assert "TWILIO_AUTH_TOKEN" in logged, "the secret's NAME should be logged -- it is a reference, not a secret"


def test_tenant_context_secret_optional_does_not_log_the_value(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    ctx = TenantContext(
        settings=TenantSettings(id="t_canary", slug="canary", name="Canary"),
        secrets_resolved={"STRINGEE_API_SECRET": CANARY},
    )

    assert ctx.secret_optional("STRINGEE_API_SECRET") == CANARY

    assert CANARY not in _logged_text(caplog), "secret_optional() logged the value"


def test_a_missing_optional_secret_still_says_so(caplog: pytest.LogCaptureFixture) -> None:
    """Safety is not the only requirement: `secret_optional` returning None is
    a real diagnostic outcome ("the credential is not configured"), and an
    event that omitted it would be safe and useless."""
    caplog.set_level(logging.DEBUG)
    ctx = TenantContext(settings=TenantSettings(id="t_canary", slug="canary", name="Canary"))

    assert ctx.secret_optional("NOT_CONFIGURED_ANYWHERE") is None

    assert "NOT_CONFIGURED_ANYWHERE" in _logged_text(caplog)
