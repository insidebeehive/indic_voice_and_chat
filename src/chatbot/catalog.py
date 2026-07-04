"""Standard CRM tool catalog for betting-platform chatbot integrations.

Each entry maps a tool name to its description, JSON-schema-style parameter
spec (with source annotations), and a default URL path template.  Tenants seed
these via POST /chat/tools/from-catalog — they supply only a base URL and auth
token; the catalog supplies everything else.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Player tools — all params injected from chat session (source="session")
# ---------------------------------------------------------------------------

PLAYER_TOOLS: dict[str, dict] = {
    "get_player_wallet": {
        "description": (
            "Get the player's wallet balances: real-money balance, bonus balance, "
            "total available balance, account currency, and any pending withdrawal "
            "amounts or statuses."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/wallet",
        "method": "GET",
    },
    "get_player_transactions": {
        "description": (
            "Get the player's transaction history: deposits, withdrawals, casino "
            "credits/debits, sports credits/debits. Supports filtering by type and "
            "date range via query params. Use to answer questions like 'did my "
            "deposit go through?' or 'show my recent withdrawals'."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
            "type":    {"type": "string", "source": "llm",
                        "description": "Filter: deposit | withdrawal | casino | sports | all (default: all)"},
            "limit":   {"type": "integer", "source": "llm",
                        "description": "Max records to return (default: 20)"},
        },
        "default_path": "/players/{user_id}/transactions",
        "method": "GET",
    },
    "get_player_bets": {
        "description": (
            "Get the player's bet slip data: open/pending bets, settled bets, "
            "most recent bet result, weekly P&L, and live cashout valuation for "
            "open bets. Use for questions about active bets, bet history, winnings."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
            "status":  {"type": "string", "source": "llm",
                        "description": "Filter: WON | LOST | SETTLED | all (default: all)"},
            "limit":   {"type": "integer", "source": "llm",
                        "description": "Max records to return (default: 20)"},
        },
        "default_path": "/players/{user_id}/bets",
        "method": "GET",
    },
    "get_player_bonuses": {
        "description": (
            "Get the player's bonus and promotion data: active bonus claims, "
            "rollover/wagering progress and amount remaining, bonus expiry dates, "
            "claim history (has welcome bonus been used?), reload/deposit bonus "
            "eligibility, referral count, and referral bonus earnings."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/bonuses",
        "method": "GET",
    },
    "get_player_profile": {
        "description": (
            "Get the player's account profile: VIP/loyalty tier and benefits, "
            "KYC verification status, submitted KYC documents, registered mobile "
            "number and email, saved bank account / UPI details, account creation "
            "date, and recent login history."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/profile",
        "method": "GET",
    },
    "get_player_responsible_gaming": {
        "description": (
            "Get the player's responsible gaming settings: self-exclusion status "
            "and end date, active deposit limits, and betting limits configured "
            "on the account. Use when the player asks about self-exclusion or "
            "their limits."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/responsible-gaming",
        "method": "GET",
    },
    "get_payment_config": {
        "description": (
            "Get the player's personalised payment configuration: the bank account "
            "number or UPI ID the player should deposit money into (varies by player "
            "tier/rating), available deposit methods (UPI, net banking, cards, wallets), "
            "withdrawal channels, minimum and maximum deposit/withdrawal limits, "
            "supported banks, blocked/unsupported banks, and withdrawal processing SLA. "
            "Call this whenever the player asks which bank account to deposit into, "
            "where to send money, what their deposit options are, or what UPI ID to use."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/payment-config",
        "method": "GET",
    },
}

# ---------------------------------------------------------------------------
# Operator tools — only operator_id injected from session
# ---------------------------------------------------------------------------

OPERATOR_TOOLS: dict[str, dict] = {
    "get_game": {
        "description": (
            "Get details about a specific game by name: rules, availability, "
            "minimum/maximum bet limits, RTP, and any active promotions for that game. "
            "Call this whenever the customer asks about a specific game by name "
            "(e.g. 'how do I play Teen Patti?', 'is Andar Bahar available?', "
            "'what are the limits for roulette?')."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
            "name": {"type": "string", "source": "llm",
                     "description": "Name of the game the customer is asking about"},
        },
        "default_path": "/operators/{operator_id}/games",
        "method": "GET",
    },
    "get_game_providers": {
        "description": (
            "Get the list of game providers available on the platform along with "
            "the game count per provider. Use this when the customer asks general "
            "questions about available games or providers without naming a specific "
            "game (e.g. 'what games do you have?', 'which providers are available?', "
            "'how many games are there?')."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
        },
        "default_path": "/operators/{operator_id}/providers",
        "method": "GET",
    },
    "get_operator_games_config": {
        "description": (
            "Get the operator's product and games configuration: which casino game "
            "providers/aggregators are enabled, available sports and leagues, whether "
            "live casino is enabled, Matka/lottery/virtual-sports availability, "
            "in-play betting support, and cashout availability on sports bets. "
            "Call this whenever the customer asks which games are available, what "
            "game types the platform offers, whether a specific category (live casino, "
            "sports, lottery, Matka) is available, or asks for a list of games."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
        },
        "default_path": "/operators/{operator_id}/games-config",
        "method": "GET",
    },
    "get_operator_promotions": {
        "description": (
            "Get the operator's current promotions and bonus configuration: "
            "active promotions list, welcome/first-deposit bonus details and "
            "wagering requirements, cashback or losing bonus config, referral "
            "program details, VIP/loyalty tier definitions and per-tier benefits."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
        },
        "default_path": "/operators/{operator_id}/promotions",
        "method": "GET",
    },
    "get_operator_platform_config": {
        "description": (
            "Get the operator's platform settings: supported currencies, "
            "available languages, timezone, minimum player age, customer support "
            "contact details (phone, email, WhatsApp, chat) and support hours, "
            "mobile app availability, KYC document requirements, geographic "
            "restrictions, and the operator/brand profile."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
        },
        "default_path": "/operators/{operator_id}/platform-config",
        "method": "GET",
    },
}

ALL_TOOLS: dict[str, dict] = {**PLAYER_TOOLS, **OPERATOR_TOOLS}
