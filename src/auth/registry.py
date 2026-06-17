"""Per-tenant runtime registries.

The framework holds these singletons per tenant for the life of the process:

- ``TenantProviders``  cached STT/LLM/TTS/telephony/vector-store client per tenant
- ``RetrieverRegistry``  one HybridRetriever per tenant (FAISS dir + BM25 index)
- ``DNDRegistry``  one DND filter + calling-hours policy per tenant
- ``WebhookRegistry``  one WebhookManager per tenant
- ``ChatChannelRegistry``  one IChatChannel per tenant for WhatsApp handoff
- ``SessionStoreRegistry``  one tenant-namespaced SessionStore per tenant
- ``CRMRegistry``  one ICRMClient per tenant

A single ``TenantRuntimeRegistry`` ties them together. The registries are
intentionally dumb (just dicts + lazy factories) — orchestration logic
goes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.auth.context import TenantContext
from src.config_tenant import merge_provider_config


# --- Provider clients ---------------------------------------------------


@dataclass
class TenantProviders:
    """Lazy provider client cache keyed by ``(tenant_id, layer)``.

    The factory functions accept a config dict and return the adapter
    instance. We pre-merge tenant overrides into the global defaults and
    pass the resolved API key in.
    """

    global_defaults: dict[str, Any]   # {"stt": {...}, "llm": {...}, ...}
    stt_factory: Callable[[dict[str, Any]], Any]
    llm_factory: Callable[[dict[str, Any]], Any]
    tts_factory: Callable[[dict[str, Any]], Any]
    telephony_factory: Callable[[dict[str, Any]], Any]
    vector_store_factory: Callable[[dict[str, Any]], Any]
    base_vector_path: Path = Path("data/faiss")
    _cache: dict[tuple[str, str], Any] = field(default_factory=dict)

    def _config_for(self, tenant: TenantContext, layer: str) -> dict[str, Any]:
        tenant_layer = getattr(tenant.settings.pipeline, layer)
        env_key = getattr(tenant_layer, "api_key_env", None) if layer != "telephony" else None
        api_key = tenant.secret(env_key) if env_key else None
        merged = merge_provider_config(
            tenant_layer,
            self.global_defaults.get(layer, {}),
            api_key=api_key,
        )
        # Telephony has dual secrets (account_sid + auth_token), resolved for the
        # tenant's *configured* provider (a tenant may hold creds for several).
        if layer == "telephony":
            creds = tenant_layer.active_creds()
            merged["account_sid"] = tenant.secret(creds.account_sid_env)
            merged["auth_token"] = tenant.secret(creds.auth_token_env)
        # Vector store gets its own per-tenant directory.
        if layer == "vector_store":
            tenant_path = self.base_vector_path / tenant.id
            tenant_path.mkdir(parents=True, exist_ok=True)
            merged["index_path"] = str(tenant_path / "index")
        return merged

    def get_stt(self, tenant: TenantContext) -> Any:
        return self._get_or_build(tenant, "stt", self.stt_factory)

    def get_llm(self, tenant: TenantContext) -> Any:
        return self._get_or_build(tenant, "llm", self.llm_factory)

    def get_tts(self, tenant: TenantContext) -> Any:
        return self._get_or_build(tenant, "tts", self.tts_factory)

    def get_telephony(self, tenant: TenantContext) -> Any:
        return self._get_or_build(tenant, "telephony", self.telephony_factory)

    def get_vector_store(self, tenant: TenantContext) -> Any:
        return self._get_or_build(tenant, "vector_store", self.vector_store_factory)

    def _get_or_build(self, tenant: TenantContext, layer: str, factory) -> Any:
        key = (tenant.id, layer)
        if key in self._cache:
            return self._cache[key]
        cfg = self._config_for(tenant, layer)
        client = factory(cfg)
        self._cache[key] = client
        return client

    def evict(self, tenant_id: Optional[str] = None) -> None:
        """Drop cached clients so a config/key update takes effect. ``tenant_id``
        None drops everything (e.g. on a full resolver reload)."""
        if tenant_id is None:
            self._cache.clear()
            return
        for key in [k for k in self._cache if k[0] == tenant_id]:
            del self._cache[key]


# --- Generic per-tenant registry helper --------------------------------


class _PerTenantRegistry:
    """Generic ``tenant_id -> instance`` cache with a lazy factory."""

    def __init__(self, factory: Callable[[TenantContext], Any]) -> None:
        self._factory = factory
        self._items: dict[str, Any] = {}

    def get(self, tenant: TenantContext) -> Any:
        if tenant.id not in self._items:
            self._items[tenant.id] = self._factory(tenant)
        return self._items[tenant.id]

    def has(self, tenant_id: str) -> bool:
        return tenant_id in self._items

    def evict(self, tenant_id: str) -> None:
        self._items.pop(tenant_id, None)

    def clear(self) -> None:
        self._items.clear()

    def items(self) -> dict[str, Any]:
        return dict(self._items)


# --- Top-level runtime registry ----------------------------------------


@dataclass
class TenantRuntimeRegistry:
    providers: TenantProviders
    retrievers: _PerTenantRegistry
    dnd: _PerTenantRegistry
    schedulers: _PerTenantRegistry
    webhooks: _PerTenantRegistry
    chat_channels: _PerTenantRegistry
    session_stores: _PerTenantRegistry
    crms: _PerTenantRegistry

    def _subregistries(self) -> tuple:
        return (
            self.retrievers, self.dnd, self.schedulers, self.webhooks,
            self.chat_channels, self.session_stores, self.crms,
        )

    def evict_tenant(self, tenant_id: str) -> None:
        self.providers.evict(tenant_id)
        for reg in self._subregistries():
            reg.evict(tenant_id)

    def evict_all(self) -> None:
        """Drop every cached per-tenant instance (providers + sub-registries) —
        wired to the resolver's on_reload so a config/key update is picked up."""
        self.providers.evict(None)
        for reg in self._subregistries():
            reg.clear()


def make_per_tenant_registry(factory: Callable[[TenantContext], Any]) -> _PerTenantRegistry:
    return _PerTenantRegistry(factory)
