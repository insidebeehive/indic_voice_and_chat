"""Encryption-at-rest for per-tenant secrets.

Only **telephony** provider keys are stored per tenant (STT/LLM/TTS/S2S use
shared master keys from the platform env). Those telephony keys are encrypted
with Fernet (AES-128-CBC + HMAC) before they touch the database and decrypted
only when a call needs them.

The Fernet key comes from the env var ``VOX_SECRET_KEY`` (a urlsafe-base64
32-byte Fernet key — generate one with ``Fernet.generate_key()``). It is
required to boot once any encrypted secret exists; rotating it orphans every
stored secret, so treat it like a master credential.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)

VOX_SECRET_KEY_ENV = "VOX_SECRET_KEY"


class SecretsError(RuntimeError):
    """Raised when the master key is missing/invalid or a token can't be decrypted."""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = os.environ.get(VOX_SECRET_KEY_ENV)
    if not key:
        raise SecretsError(
            f"{VOX_SECRET_KEY_ENV} is not set — required to encrypt/decrypt tenant "
            f"secrets. Generate one with: "
            f"python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    try:
        fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except (ValueError, TypeError) as e:
        raise SecretsError(f"{VOX_SECRET_KEY_ENV} is not a valid Fernet key: {e}") from e
    # Fires once per process (lru_cache maxsize=1) -- never the key itself,
    # only its length, so an operator can confirm a key IS configured and
    # roughly matches expectations without it ever leaving os.environ.
    from src.utils.logging import debug_event
    debug_event(log, "secrets fernet_key loaded", key_len=len(key))
    return fernet


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage. Returns a urlsafe-base64 token (str)."""
    token = _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")
    # Boundary: every telephony secret written to tenant_secrets crosses here.
    # Lengths only -- never plaintext or the encrypted token itself (the
    # token is not the plaintext, but it is a working, decryptable credential
    # and is treated with the same care).
    from src.utils.logging import debug_event
    debug_event(log, "secrets encrypt response", plaintext_len=len(plaintext), token_len=len(token))
    return token


def decrypt(token: str) -> str:
    """Decrypt a stored secret token back to plaintext."""
    try:
        value = _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise SecretsError(
            "could not decrypt a stored secret — the VOX_SECRET_KEY likely changed "
            "since it was encrypted"
        ) from e
    # Boundary: every per-tenant telephony secret decrypted at resolver-reload
    # time crosses here. Length only, never the decrypted value -- the caller
    # (db_resolver.py's tenant_context_from_row) already logs a decrypt
    # FAILURE at ERROR level with the secret name; this covers the success
    # side, which previously left no trace at any level.
    from src.utils.logging import debug_event
    debug_event(log, "secrets decrypt response", token_len=len(token), value_len=len(value))
    return value


def has_key() -> bool:
    """True if a master key is configured (without raising). Used to fail a
    register request early when telephony keys are supplied but no key is set."""
    return bool(os.environ.get(VOX_SECRET_KEY_ENV))


def reset_cache_for_tests() -> None:
    """Drop the cached Fernet so a test can change VOX_SECRET_KEY."""
    _fernet.cache_clear()


def generate_key() -> str:
    """Mint a fresh Fernet key (for setup/scripts)."""
    return Fernet.generate_key().decode("ascii")
