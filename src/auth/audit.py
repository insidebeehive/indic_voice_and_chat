"""Shared audit-logging helpers.

Currently provides only ``token_fingerprint``, a one-way, non-reversible
fingerprint safe to place in log lines wherever a raw token/id would
otherwise leak PII (player UUIDs, session ids, phone numbers, etc.).

A future PR will extend this module with a shared ``log_denied(...)``
audit-logging helper (for auth/tenant-resolution rejections) and
rejection-log suppression logic (to avoid log-flooding on repeated
denials from the same source). Deliberately not built yet — this PR is
narrow: just the fingerprint primitive and its use in
``src/api/external_chat.py``.
"""

import hashlib


def token_fingerprint(value: str, *, domain: str = "vox-logfp-v1") -> str:
    """Returns a 12-hex-char fingerprint of `value`, safe to log.

    NOT the same hash used for credential verification (see
    src/auth/context.py's hash_api_token) — deliberately domain-separated
    via the `domain` prefix so this fingerprint can never be confused with,
    or substituted for, the actual stored credential-verification hash.
    Publishing a prefix of the real verification hash would be a partial
    disclosure of the stored secret; this is a distinct derivation.

    Use a distinct `domain` string per value-type (e.g. "vox-logfp-sid-v1"
    for session ids, "vox-logfp-tel-v1" for phone numbers) so a fingerprint
    computed in one context can never be cross-referenced against another.
    """
    return hashlib.sha256(f"{domain}:{value}".encode("utf-8")).hexdigest()[:12]
