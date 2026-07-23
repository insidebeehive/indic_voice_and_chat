"""FastAPI routes — central router that aggregates all sub-routers."""

from fastapi import APIRouter

from src.api import (
    benchmarks,
    calls,
    campaigns,
    catalog,
    chat,
    chat_tools,
    config_routes,
    conversations,
    crm_kb,
    crms,
    external_chat,
    knowledge,
    sessions,
    softphone,
    telephony_crm,
    telephony_hooks,
    tenants,
    webhooks_routes,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions.router)
api_router.include_router(tenants.router)
api_router.include_router(catalog.router)
api_router.include_router(campaigns.router)
api_router.include_router(calls.router)
api_router.include_router(softphone.router)
api_router.include_router(config_routes.router)
api_router.include_router(conversations.router)
api_router.include_router(crms.router)
api_router.include_router(crm_kb.router)
api_router.include_router(knowledge.router)
api_router.include_router(webhooks_routes.router)
api_router.include_router(benchmarks.router)
api_router.include_router(chat.router)
api_router.include_router(chat_tools.router)
api_router.include_router(external_chat.router)
api_router.include_router(telephony_hooks.router)
api_router.include_router(telephony_crm.router)
