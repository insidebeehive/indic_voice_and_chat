"""Provider → telephony answer-webhook route path (under ``/api/v1/telephony/``).

Shared by the dev-console ``place-call`` and the campaign ``call_lead`` paths so
the OUTBOUND answer URL is slug-scoped consistently: ``{base}/{path}/{slug}``.
For a call WE place the tenant is known, so the answer webhook resolves by slug
(``tenant_from_slug``) instead of reverse-resolving our own caller-ID — which
would require the number to be registered in ``tenant_phone_numbers``.
"""

from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit

ANSWER_PATHS: dict[str, str] = {
    "twilio": "twilio/voice",
    "exotel": "exotel/voice",
    "stringee": "stringee/answer",
}
"""Providers that have a slug-scoped answer route. A provider absent here gets a
bare base URL (unchanged legacy behaviour)."""


def answer_url_for(tenant, provider: str, webhook_base: str) -> str:
    """Builds the outbound answer URL for ``provider`` (e.g. ``stringee``,
    ``exotel``, ``twilio``) using ``ANSWER_PATHS`` + ``tenant.slug`` — the same
    base-URL construction as the ``calls.py`` call site (tolerant ``.get()``,
    falling back to the bare ``webhook_base`` for a provider not in the map) —
    then layers in an optional per-tenant, per-provider credential:

    - ``stringee``: appends ``?vt=<token>`` (URL-encoded) if the tenant has
      ``webhook:stringee_path_token`` configured.
    - ``exotel``: injects ``user:pass@`` into the URL's netloc if the tenant has
      BOTH ``webhook:exotel_basic_user`` and ``webhook:exotel_basic_password``
      configured. An incomplete pair (only one of the two set) is left as a
      no-op — it must never produce a partially-credentialed URL.
    - ``twilio`` (and any other/unknown provider): no change — twilio relies on
      signature verification, not a URL-embedded credential.

    A no-op / backward-compatible passthrough when the tenant hasn't configured
    the relevant secret(s): returns exactly the base ``ANSWER_PATHS``-built URL.

    SECURITY: the returned string can embed credentials (a ``vt=`` token or a
    ``user:pass@`` netloc). Callers must never log it in full.
    """
    base = (webhook_base or "").rstrip("/")
    path = ANSWER_PATHS.get(provider)
    url = f"{base}/{path}/{tenant.slug}" if path else base

    sr = getattr(tenant, "secrets_resolved", None) or {}

    if provider == "stringee":
        token = sr.get("webhook:stringee_path_token")
        if token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}vt={quote(token, safe='')}"
    elif provider == "exotel":
        user = sr.get("webhook:exotel_basic_user")
        password = sr.get("webhook:exotel_basic_password")
        if user and password:
            parts = urlsplit(url)
            # An empty netloc means webhook_base itself was empty/None/malformed —
            # there's no actual host to embed credentials into, so injecting
            # `user:pass@` here would just produce a nonsensical, dangling
            # `//user:pass@/...` URL. Leave it as a no-op in that case.
            if parts.netloc:
                netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.netloc}"
                url = urlunsplit(
                    (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
                )

    return url
