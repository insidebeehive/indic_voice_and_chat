"""Unit tests for ``answer_url_for`` — the tenant-aware answer-URL builder.

Standalone: ``answer_url_for`` is not yet wired into any call site (that's
later work), so these tests exercise it in isolation against a lightweight
tenant stub rather than the real ``TenantContext``.
"""

from types import SimpleNamespace

from src.api.answer_paths import ANSWER_PATHS, answer_url_for


def make_tenant(slug: str = "acme", secrets_resolved: dict | None = None):
    kwargs = {"slug": slug}
    if secrets_resolved is not None:
        kwargs["secrets_resolved"] = secrets_resolved
    return SimpleNamespace(**kwargs)


def legacy_url(webhook_base: str, provider: str, slug: str) -> str:
    """Reproduces calls.py's existing plain construction for comparison."""
    base = webhook_base.rstrip("/")
    path = ANSWER_PATHS.get(provider)
    return f"{base}/{path}/{slug}" if path else base


class TestUnconfiguredTenant:
    def test_stringee_matches_legacy_construction(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "stringee", "https://example.com")
        assert url == legacy_url("https://example.com", "stringee", "acme")

    def test_exotel_matches_legacy_construction(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "exotel", "https://example.com")
        assert url == legacy_url("https://example.com", "exotel", "acme")

    def test_twilio_matches_legacy_construction(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "twilio", "https://example.com")
        assert url == legacy_url("https://example.com", "twilio", "acme")

    def test_missing_secrets_resolved_attribute_entirely(self):
        # No secrets_resolved attribute at all — getattr(..., None) or {} path.
        tenant = SimpleNamespace(slug="acme")
        for provider in ("stringee", "exotel", "twilio"):
            url = answer_url_for(tenant, provider, "https://example.com")
            assert url == legacy_url("https://example.com", provider, "acme")


class TestStringeeToken:
    def test_appends_vt_query_param(self):
        tenant = make_tenant(secrets_resolved={"webhook:stringee_path_token": "abc123"})
        url = answer_url_for(tenant, "stringee", "https://example.com")
        assert url == "https://example.com/stringee/answer/acme?vt=abc123"

    def test_url_encodes_special_characters_in_token(self):
        tenant = make_tenant(
            secrets_resolved={"webhook:stringee_path_token": "a b/c&d=e?f"}
        )
        url = answer_url_for(tenant, "stringee", "https://example.com")
        assert url == "https://example.com/stringee/answer/acme?vt=a%20b%2Fc%26d%3De%3Ff"


class TestExotelCredentials:
    def test_both_user_and_password_injects_netloc(self):
        tenant = make_tenant(
            secrets_resolved={
                "webhook:exotel_basic_user": "user1",
                "webhook:exotel_basic_password": "pass1",
            }
        )
        url = answer_url_for(tenant, "exotel", "https://example.com")
        assert url == "https://user1:pass1@example.com/exotel/voice/acme"

    def test_only_user_leaves_url_unchanged(self):
        tenant = make_tenant(
            secrets_resolved={"webhook:exotel_basic_user": "user1"}
        )
        url = answer_url_for(tenant, "exotel", "https://example.com")
        assert url == legacy_url("https://example.com", "exotel", "acme")

    def test_only_password_leaves_url_unchanged(self):
        tenant = make_tenant(
            secrets_resolved={"webhook:exotel_basic_password": "pass1"}
        )
        url = answer_url_for(tenant, "exotel", "https://example.com")
        assert url == legacy_url("https://example.com", "exotel", "acme")

    def test_empty_webhook_base_does_not_inject_dangling_credentials(self):
        # An empty webhook_base produces an empty netloc after urlsplit — there's
        # no host to embed credentials into, so injection must be skipped rather
        # than producing a malformed `//user:pass@/...` URL.
        tenant = make_tenant(
            secrets_resolved={
                "webhook:exotel_basic_user": "user1",
                "webhook:exotel_basic_password": "pass1",
            }
        )
        url = answer_url_for(tenant, "exotel", "")
        assert "user1:pass1@" not in url
        assert "//user1:pass1@/" not in url
        assert url == legacy_url("", "exotel", "acme")

    def test_none_webhook_base_does_not_inject_dangling_credentials(self):
        # `webhook_base or ""` already tolerates None (same as the unconfigured
        # tests' implicit coverage via the base construction), so this exercises
        # the same empty-netloc no-op path via a None input.
        tenant = make_tenant(
            secrets_resolved={
                "webhook:exotel_basic_user": "user1",
                "webhook:exotel_basic_password": "pass1",
            }
        )
        url = answer_url_for(tenant, "exotel", None)
        assert "user1:pass1@" not in url
        assert "//user1:pass1@/" not in url


class TestTwilioIgnoresOtherProvidersSecrets:
    def test_stringee_and_exotel_secrets_present_no_effect(self):
        tenant = make_tenant(
            secrets_resolved={
                "webhook:stringee_path_token": "abc123",
                "webhook:exotel_basic_user": "user1",
                "webhook:exotel_basic_password": "pass1",
            }
        )
        url = answer_url_for(tenant, "twilio", "https://example.com")
        assert url == legacy_url("https://example.com", "twilio", "acme")


class TestUnknownProvider:
    def test_falls_back_to_bare_webhook_base(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "carrier-pigeon", "https://example.com")
        assert url == "https://example.com"

    def test_unknown_provider_with_secrets_still_bare(self):
        tenant = make_tenant(
            secrets_resolved={
                "webhook:stringee_path_token": "abc123",
                "webhook:exotel_basic_user": "user1",
                "webhook:exotel_basic_password": "pass1",
            }
        )
        url = answer_url_for(tenant, "carrier-pigeon", "https://example.com")
        assert url == "https://example.com"


class TestTrailingSlash:
    def test_no_double_slash_with_trailing_slash_base(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "stringee", "https://example.com/")
        assert url == "https://example.com/stringee/answer/acme"
        assert "//stringee" not in url

    def test_no_double_slash_bare_fallback_with_trailing_slash(self):
        tenant = make_tenant(secrets_resolved={})
        url = answer_url_for(tenant, "unknown", "https://example.com/")
        assert url == "https://example.com"
