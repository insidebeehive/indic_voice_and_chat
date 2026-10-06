"""Mock CRM server for local chatbot development and demos.

Implements all endpoints from src/chatbot/catalog.py with realistic fixture
data for a fictional Indian betting platform.

Usage:
    python -m tools.mock_crm              # runs on :9000
    python -m tools.mock_crm --port 9001  # custom port

Seed tools via:
    POST /chat/tools/from-catalog
    { "crm_base_url": "http://localhost:9000", "auth_type": "bearer", "auth_token": "mock" }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Path, Query
from typing import Optional

app = FastAPI(title="Mock CRM", version="1.0.0")

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

PLAYERS: dict[str, dict] = {
    "player_001": {
        "wallet": {
            "real_balance": 4250.75,
            "bonus_balance": 500.00,
            "total_available": 4750.75,
            "currency": "INR",
            "pending_withdrawal": {"amount": 1000.00, "status": "processing"},
        },
        "transactions": [
            {"id": "txn_001", "type": "deposit",    "amount": 2000.00, "status": "success",    "timestamp": "2026-06-20T10:30:00Z", "method": "UPI"},
            {"id": "txn_002", "type": "casino",     "amount": -350.00, "status": "settled",    "timestamp": "2026-06-20T11:15:00Z", "game": "Teen Patti"},
            {"id": "txn_003", "type": "sports",     "amount": 600.00,  "status": "settled",    "timestamp": "2026-06-20T14:00:00Z", "description": "IPL bet win"},
            {"id": "txn_004", "type": "withdrawal", "amount": 1000.00, "status": "processing", "timestamp": "2026-06-20T16:00:00Z", "method": "Bank Transfer"},
        ],
        "latest_deposit_order": {
            "order_id": "a1b2c3d4-e29b-41d4-a716-446655440000",
            "external_transaction_id": None,
            "amount": 1500.00,
            "currency": "INR",
            "pgs_status": "PGS_FAILED",
            "status_bucket": "failed",
            "created_at": "2026-06-21T14:30:00+05:30",
        },
        "bets": [
            {"id": "bet_001", "sport": "Cricket", "match": "MI vs CSK", "selection": "MI Win",
             "stake": 500.00, "odds": 1.85, "status": "open", "cashout_value": 480.00,
             "placed_at": "2026-06-21T09:00:00Z"},
            {"id": "bet_002", "sport": "Cricket", "match": "IPL Final", "selection": "RCB Win",
             "stake": 300.00, "odds": 2.10, "returns": 630.00, "status": "won",
             "settled_at": "2026-06-20T14:00:00Z"},
        ],
        "pnl": {"this_week": 250.00, "this_month": -150.00},
        "bonuses": {
            "active": [
                {"name": "Welcome Bonus", "amount": 500.00, "wagering_required": 2500.00,
                 "wagering_completed": 800.00, "expires_at": "2026-07-05T00:00:00Z"},
            ],
            "welcome_bonus_claimed": True,
            "reload_eligible": True,
            "referrals": {"count": 3, "earnings": 300.00},
        },
        "profile": {
            "vip_tier": "Silver",
            "vip_benefits": ["Dedicated support", "Weekly cashback 5%", "Faster withdrawals"],
            "kyc_status": "verified",
            "kyc_documents": ["Aadhaar", "PAN"],
            "mobile": "+91-98765-43210",
            "email": "player001@example.com",
            "bank_saved": {"bank": "HDFC", "account_last4": "7890", "upi": "player001@upi"},
            "created_at": "2025-03-15T00:00:00Z",
            "last_login": "2026-06-21T08:45:00Z",
        },
        "responsible_gaming": {
            "self_excluded": False,
            "deposit_limit": {"daily": 10000, "weekly": 50000, "currency": "INR"},
            "bet_limit": None,
        },
    },
    "player_002": {
        "wallet": {
            "real_balance": 125.00,
            "bonus_balance": 0.00,
            "total_available": 125.00,
            "currency": "INR",
            "pending_withdrawal": None,
        },
        "transactions": [
            {"id": "txn_010", "type": "deposit", "amount": 500.00, "status": "success",
             "timestamp": "2026-06-18T09:00:00Z", "method": "Net Banking"},
            {"id": "txn_011", "type": "casino",  "amount": -375.00, "status": "settled",
             "timestamp": "2026-06-18T10:30:00Z", "game": "Roulette"},
        ],
        "bets": [],
        "pnl": {"this_week": -375.00, "this_month": -375.00},
        "bonuses": {
            "active": [],
            "welcome_bonus_claimed": True,
            "reload_eligible": False,
            "referrals": {"count": 0, "earnings": 0.00},
        },
        "profile": {
            "vip_tier": "Bronze",
            "vip_benefits": ["Standard support"],
            "kyc_status": "pending",
            "kyc_documents": ["Aadhaar"],
            "mobile": "+91-87654-32109",
            "email": "player002@example.com",
            "bank_saved": None,
            "created_at": "2026-05-01T00:00:00Z",
            "last_login": "2026-06-21T07:00:00Z",
        },
        "responsible_gaming": {
            "self_excluded": False,
            "deposit_limit": None,
            "bet_limit": None,
        },
    },
}

OPERATORS: dict[str, dict] = {
    "op_demo": {
        "payment_config": {
            "deposit_methods": ["UPI", "Net Banking", "Debit Card", "Paytm", "PhonePe"],
            "withdrawal_channels": ["Bank Transfer", "UPI"],
            "deposit_limits": {"min": 100, "max": 100000, "currency": "INR"},
            "withdrawal_limits": {"min": 500, "max": 50000, "currency": "INR"},
            "supported_banks": ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank"],
            "blocked_banks": ["PMJDY accounts"],
            "upi_supported": True,
            "withdrawal_processing_time": "24-48 hours (business days)",
        },
        "games_config": {
            "casino": {"enabled": True, "under_maintenance": False, "default_url": None},
            "sports": {"enabled": True, "external": False, "internal": True,
                       "provider": "INTERNAL", "under_maintenance": False},
            "sports_exchange": {"enabled": False},
            "matka": {"enabled": True, "under_maintenance": False},
            "lottery": {"enabled": False},
        },
        "promotions": {
            "active_promotions": [
                {"name": "Welcome Bonus", "type": "first_deposit",
                 "offer": "100% match up to ₹10,000", "wagering_requirement": "5x",
                 "min_deposit": 500, "valid_until": "2026-12-31"},
                {"name": "IPL Cashback", "type": "cashback",
                 "offer": "10% cashback on losses every Monday", "valid_until": "2026-07-31"},
                {"name": "Refer & Earn", "type": "referral",
                 "offer": "₹100 bonus per verified referral", "max_referrals": 50},
            ],
            "vip_tiers": [
                {"tier": "Bronze", "min_deposits": 0,    "benefits": ["Standard support"]},
                {"tier": "Silver", "min_deposits": 5000,  "benefits": ["5% weekly cashback", "Priority support", "Faster withdrawals"]},
                {"tier": "Gold",   "min_deposits": 25000, "benefits": ["10% weekly cashback", "Dedicated manager", "Same-day withdrawals"]},
                {"tier": "Platinum","min_deposits": 100000,"benefits": ["15% cashback", "Personal manager", "Instant withdrawals", "Event invites"]},
            ],
        },
        "platform_config": {
            "currencies": ["INR"],
            "languages": ["Hindi", "English", "Telugu", "Tamil", "Marathi", "Bengali"],
            "timezone": "Asia/Kolkata",
            "minimum_age": 18,
            "support_contacts": {
                "phone": "+91-800-123-4567",
                "email": "support@demoplatform.com",
                "whatsapp": "+91-900-123-4567",
                "live_chat": True,
            },
            "support_hours": "24x7",
            "mobile_app": {"android": True, "ios": False},
            "kyc_documents_required": ["Government photo ID (Aadhaar/PAN/Passport/Voter ID)", "Proof of address"],
            "geo_restrictions": ["Not available in Telangana, Andhra Pradesh, Sikkim for certain games"],
            "brand": "Demo Platform",
            "operator_profile": "Licensed online gaming platform serving Indian players since 2022.",
        },
        "bet_limits": [
            {"game_name": "Teen Patti", "games_master_id": "gm-101", "min_stake": 10,
             "max_stake": 50000, "max_profit": 200000, "resolved_from": "tier_override"},
            {"game_name": "Andar Bahar", "games_master_id": "gm-102", "min_stake": 5,
             "max_stake": 25000, "max_profit": 100000, "resolved_from": "default"},
            # No limit configured at any tier for this game -- exists, but
            # every limit field (and resolved_from) comes back null, per the
            # shipped contract's "no limit configured" case.
            {"game_name": "Roulette", "games_master_id": "gm-103", "min_stake": None,
             "max_stake": None, "max_profit": None, "resolved_from": None},
        ],
        "matka_markets": {
            # Keyed lowercase for partial-match lookup (see _match_matka_market).
            "kalyan": {
                "market_name": "Kalyan",
                "closed_weekdays": {"Sunday"},
                "upcoming_closures": [
                    {"date": "2026-08-17", "weekday": "Sunday"},
                    {"date": "2026-08-24", "weekday": "Sunday"},
                ],
            },
            "milan day": {
                "market_name": "Milan Day",
                "closed_weekdays": {"Sunday"},
                "upcoming_closures": [
                    {"date": "2026-08-17", "weekday": "Sunday"},
                    {"date": "2026-08-24", "weekday": "Sunday"},
                ],
            },
        },
    },
    # Matka sold to selected operators only, so the holiday-schedule route
    # 403s when it's disabled. Without an operator that actually has it off,
    # that branch is unreachable here: _get_operator falls back to op_demo
    # (matka enabled) for any unknown id, so the 403 the tool description
    # tells the model how to handle could never be produced by this mock.
    # Only the fields the Matka guard reads are set.
    "op_matka_off": {
        "games_config": {
            "casino": {"enabled": True, "under_maintenance": False, "default_url": None},
            "matka": {"enabled": False, "under_maintenance": False},
        },
    },
}

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(authorization: Optional[str]) -> None:
    # In mock mode, accept any bearer token (or no token for easy testing).
    pass


# ---------------------------------------------------------------------------
# Player endpoints
# ---------------------------------------------------------------------------

def _get_player(user_id: str) -> dict:
    if user_id not in PLAYERS:
        raise HTTPException(status_code=404, detail=f"player '{user_id}' not found in mock data")
    return PLAYERS[user_id]


@app.get("/players/{user_id}/wallet")
async def get_player_wallet(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_player(user_id)["wallet"]


@app.get("/players/{user_id}/transactions")
async def get_player_transactions(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(10),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    txns = _get_player(user_id)["transactions"]
    if type and type != "all":
        txns = [t for t in txns if t["type"] == type]
    return {"transactions": txns[:limit], "total": len(txns)}


@app.get("/players/{user_id}/latest-deposit-order")
async def get_player_latest_deposit_order(
    user_id: str = Path(...),
    # The shipped contract sends operator_id as the "operatorid" HEADER (see
    # docs/crm-api-contract.md's Auth section), not a query param -- this
    # endpoint takes no query params at all.
    operatorid: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    order = _get_player(user_id).get("latest_deposit_order")
    return {
        "operator_id": operatorid or "op_demo",
        "user_id": user_id,
        "status": "found" if order else "no_recent_deposit",
        "lookback_days": 7,
        "order": order,
    }


@app.get("/players/{user_id}/bets")
async def get_player_bets(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(10),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    player = _get_player(user_id)
    bets = player["bets"]
    if status and status != "all":
        bets = [b for b in bets if b["status"] == status]
    return {"bets": bets[:limit], "pnl": player["pnl"]}


@app.get("/players/{user_id}/bonuses")
async def get_player_bonuses(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_player(user_id)["bonuses"]


@app.get("/players/{user_id}/profile")
async def get_player_profile(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_player(user_id)["profile"]


@app.get("/players/{user_id}/responsible-gaming")
async def get_player_responsible_gaming(
    user_id: str = Path(...),
    operator_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_player(user_id)["responsible_gaming"]


# ---------------------------------------------------------------------------
# Operator endpoints
# ---------------------------------------------------------------------------

def _get_operator(operator_id: str) -> dict:
    # Fall back to demo operator if the real one isn't in fixtures.
    return OPERATORS.get(operator_id) or OPERATORS["op_demo"]


@app.get("/operators/{operator_id}/payment-config")
async def get_operator_payment_config(
    operator_id: str = Path(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_operator(operator_id)["payment_config"]


@app.get("/operators/{operator_id}/games-config")
async def get_operator_games_config(
    operator_id: str = Path(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return {"operator_id": operator_id, **_get_operator(operator_id)["games_config"]}


@app.get("/operators/{operator_id}/promotions")
async def get_operator_promotions(
    operator_id: str = Path(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_operator(operator_id)["promotions"]


@app.get("/operators/{operator_id}/platform-config")
async def get_operator_platform_config(
    operator_id: str = Path(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    return _get_operator(operator_id)["platform_config"]


def _match_bet_limit_games(operator_id: str, name: str) -> list[dict]:
    """Partial, case-insensitive match against the operator's casino games —
    mirrors the shipped CRM's 'name' partial-match semantics, including that
    it can legitimately match more than one game."""
    name_lower = name.lower().strip()
    return [
        g for g in _get_operator(operator_id).get("bet_limits", [])
        if name_lower in g["game_name"].lower()
    ]


@app.get("/operators/{operator_id}/players/{user_id}/bet-limit")
async def get_bet_limit(
    operator_id: str = Path(...),
    user_id: str = Path(...),
    name: str = Query(...),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    _get_player(user_id)  # 404s if the player itself doesn't exist
    if not name.strip():
        raise HTTPException(status_code=400, detail="'name' must not be blank")
    matched = _match_bet_limit_games(operator_id, name)
    if not matched:
        # Per the shipped contract: no casino game matched -- including a
        # Matka market name, since Matka isn't in this table at all.
        raise HTTPException(status_code=404, detail=f"no casino game matched '{name}'")
    return {"operator_id": operator_id, "user_id": user_id, "games": matched}


def _match_matka_market(operator_id: str, market: str) -> Optional[dict]:
    """Partial, case-insensitive match against the operator's Matka markets.
    An ambiguous match (more than one hit) is treated the same as no match
    at all -- per the shipped contract, an ambiguous market name returns
    status: "not_found" rather than guessing between matches."""
    market_lower = market.lower().strip()
    matches = [
        m for key, m in _get_operator(operator_id).get("matka_markets", {}).items()
        if market_lower in key
    ]
    return matches[0] if len(matches) == 1 else None


@app.get("/operators/{operator_id}/matka/holiday-schedule")
async def get_market_holiday_schedule(
    operator_id: str = Path(...),
    market: str = Query(...),
    date: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    games_config = _get_operator(operator_id).get("games_config", {})
    if not games_config.get("matka", {}).get("enabled", False):
        raise HTTPException(status_code=403, detail="Matka is not enabled for this operator")

    if date is None:
        query_date = _dt.datetime.now(_dt.timezone.utc).date()
    else:
        try:
            query_date = _dt.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"malformed date '{date}', expected YYYY-MM-DD") from None
    date_str = query_date.isoformat()

    matched = _match_matka_market(operator_id, market)
    if matched is None:
        return {
            "operator_id": operator_id, "market_name": None, "status": "not_found",
            "date": date_str, "is_closed": None, "closure_reason": None,
            "upcoming_closures": [], "closure_source": None,
        }

    weekday = query_date.strftime("%A")
    is_closed = weekday in matched["closed_weekdays"]
    return {
        "operator_id": operator_id,
        "market_name": matched["market_name"],
        "status": "found",
        "date": date_str,
        "is_closed": is_closed,
        "closure_reason": f"Closed every {weekday}" if is_closed else None,
        "upcoming_closures": matched["upcoming_closures"],
        "closure_source": "weekly_schedule",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock CRM server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
