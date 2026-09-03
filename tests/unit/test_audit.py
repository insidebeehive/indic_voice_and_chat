"""Unit tests for src.auth.audit.token_fingerprint."""

from __future__ import annotations

import hashlib
import re

from src.auth.audit import token_fingerprint

_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def test_deterministic():
    assert token_fingerprint("player-uuid-1") == token_fingerprint("player-uuid-1")


def test_different_inputs_produce_different_outputs():
    assert token_fingerprint("player-uuid-1") != token_fingerprint("player-uuid-2")


def test_output_is_twelve_hex_chars():
    fp = token_fingerprint("some-value")
    assert _HEX12.match(fp), fp


def test_domain_separation_vs_unsalted_sha256():
    unsalted = hashlib.sha256("abc".encode()).hexdigest()[:12]
    assert token_fingerprint("abc") != unsalted


def test_different_domains_produce_different_fingerprints_for_same_value():
    a = token_fingerprint("same-value", domain="vox-logfp-sid-v1")
    b = token_fingerprint("same-value", domain="vox-logfp-tel-v1")
    assert a != b
