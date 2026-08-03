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
            "eligibility, referral count, and referral bonus earnings. "
            "NOTE: this does NOT return the referral code or referral link — "
            "call get_referral_code for those."
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
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/operators/{operator_id}/players/{user_id}/payment-config",
        "method": "GET",
    },
    "get_referral_code": {
        "description": (
            "Get the player's personal referral code and shareable referral link. "
            "Call this when the player asks for their referral code, referral link, "
            "invite link, or wants to refer a friend "
            "(e.g. 'mera referral code kya hai', 'invite link bhejo', "
            "'dost ko refer karna hai')."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/referral-code",
        "method": "GET",
    },
    "get_sports_open_bets": {
        "description": (
            "Get the player's currently open / pending sports bets with match details "
            "and live cashout valuation. Use when the player asks to see their open "
            "bets, active bets, or wants to know how many bets are pending "
            "(e.g. 'open bets dikhao', 'pending bet kitne hain', 'live bet status')."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
            "limit": {"type": "integer", "source": "llm",
                      "description": "Max records to return (default: 10)"},
        },
        "default_path": "/players/{user_id}/sports-bets/open",
        "method": "GET",
    },
    "get_sports_match_status": {
        "description": (
            "Get today's sports markets where the player has placed bets, along with "
            "the market result and whether the player won or lost each bet. Use when "
            "the player asks about a match result or whether their bet won or lost "
            "(e.g. 'match jeeta ya haara', 'bet result kya aaya', 'kya main jeeta', "
            "'aaj ka bet settle hua')."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
        },
        "default_path": "/players/{user_id}/sports-bets/today-results",
        "method": "GET",
    },
    "get_casino_game_history": {
        "description": (
            "Get the player's history for a specific casino game: sessions played, "
            "total wagered, total won, net P&L, and recent session results. "
            "Call this when the player asks about their performance or history in a "
            "named casino game (e.g. 'Teen Patti mein kitna jeeta', "
            "'Andar Bahar ka history dikhao', 'slots ka record kya hai'). "
            "Always extract the game name from the player's message and pass it as game_name."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
            "game_name": {"type": "string", "source": "llm",
                          "description": "Name of the casino game (e.g. Teen Patti, Andar Bahar, Roulette, Slots)"},
            "limit": {"type": "integer", "source": "llm",
                      "description": "Max session records to return (default: 10)"},
        },
        "default_path": "/players/{user_id}/casino-game-history",
        "method": "GET",
    },
    "get_matka_bids": {
        "description": (
            "Get the player's Matka bid data: open/pending bids, full bid history, "
            "settlement results per market, bid acceptance status, cancellation and "
            "refund status for cancelled markets, payout/credit status for winnings, "
            "and Matka P&L summary. "
            "Call this for ANY player Matka query about their own bids or results "
            "(e.g. 'Matka mein open bets dikhao', 'meri Matka history dikhao', "
            "'All Bids dikhao', 'kya mera bet accept hua?', "
            "'Kalyan ka result kya aaya mera bet ka?', 'mera bet history mein nahi dikh raha', "
            "'market cancel hua toh refund milega?', 'is hafte Matka mein kitna jeeta/haara?', "
            "'meri jeet credit kyu nahi hui?', 'bet accept kyu nahi hua?'). "
            "NOTE: for a market's result asked WITHOUT reference to the player's "
            "own bet (e.g. a customer just asking 'Kalyan ka result kya aaya' with "
            "no bet-status question), call get_matka_result instead."
        ),
        "parameters": {
            "user_id": {"type": "string", "source": "session",
                        "description": "Player identifier"},
            "status": {"type": "string", "source": "llm",
                       "description": "Filter: open | settled | cancelled | all (default: all)"},
            "market": {"type": "string", "source": "llm",
                       "description": "Optional: filter to a specific market name (e.g. 'Kalyan', 'Milan Day', 'Starline')"},
            "limit": {"type": "integer", "source": "llm",
                      "description": "Max records to return (default: 20)"},
        },
        "default_path": "/players/{user_id}/matka-bids",
        "method": "GET",
    },
}

# ---------------------------------------------------------------------------
# Operator tools — only operator_id injected from session
# ---------------------------------------------------------------------------

OPERATOR_TOOLS: dict[str, dict] = {
    "get_game": {
        "description": (
            "Search for games by name and return a list of matching games with details "
            "(title, provider, availability, bet limits, RTP). Returns multiple results "
            "when the name matches several variants — e.g. searching 'Andar Bahar' "
            "returns every Andar Bahar variant across all providers. "
            "Call this whenever the customer asks about a named game, wants to know "
            "how many variants exist, or wants to list games of a specific type "
            "(e.g. 'how many Andar Bahar games?', 'list all roulette variants', "
            "'what Teen Patti games are available?', 'is Andar Bahar available?', "
            "'how do I play Teen Patti?', 'what are the limits for roulette?')."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
            "name": {"type": "string", "source": "llm",
                     "description": "Name or type of game to search for (e.g. 'Andar Bahar', 'Roulette', 'Teen Patti') — returns all matching variants"},
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
            "Call this when the customer asks which game CATEGORIES or product types "
            "are available on the platform (e.g. 'do you have live casino?', "
            "'is sports betting available?', 'do you have Matka?', "
            "'what types of games are there?'). "
            "For specific named game or variant questions use get_game instead. "
            "For Matka-specific details (markets, odds, timing, bet types) use get_matka_config instead."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
        },
        "default_path": "/operators/{operator_id}/games-config",
        "method": "GET",
    },
    "get_matka_config": {
        "description": (
            "Get the operator's Matka configuration: available markets (e.g. Kalyan, "
            "Milan Day, Rajdhani Night), whether Starline and Jackpot Matka are enabled, "
            "supported bet types per market (Single, Jodi, Patti, Half/Full Sangam, "
            "SP/DP/TP, Red Bracket), payout odds for each bet type, market session timings "
            "(open bid / close bid / open result / close result times), minimum and maximum "
            "stake limits per bet type, whether Panna/Jodi charts and result history are "
            "accessible, and legacy Matka rounds availability. "
            "Call this for ANY Matka question about platform config, game types, odds, "
            "timing, or limits (e.g. 'which Matka markets are available?', "
            "'is Starline available?', 'is Jackpot Matka available?', "
            "'what bet types are supported?', 'what are the payout rates for Jodi?', "
            "'when does Kalyan close?', 'what is the minimum Matka bet?'). "
            "NOTE: this does NOT return an actual declared result number — "
            "call get_matka_result for that."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
            "market_name": {"type": "string", "source": "llm",
                            "description": "Optional: filter to a specific market (e.g. 'Kalyan', 'Milan Day', 'Starline'). Omit to get full Matka config."},
        },
        "default_path": "/operators/{operator_id}/matka-config",
        "method": "GET",
    },
    "get_matka_result": {
        "description": (
            "Get the declared/settled result for a specific Matka market and "
            "session (e.g. today's Rajdhani Day, Kalyan, Milan Day number) — "
            "independent of whether the customer has a bet placed on it, and "
            "usable even for a customer who isn't logged in with an active bet. "
            "Call this whenever the customer asks for a market's result directly "
            "(e.g. 'Rajdhani Day ka result kya aya hai', 'Kalyan ka aaj ka number "
            "kya hai', 'is market ka result declare hua kya', 'Milan Day open "
            "result kya tha'). "
            "NOTE: this does NOT know which bets a player placed or whether they "
            "won/lost — for the customer's OWN bid outcome, settlement status, or "
            "payout, call get_matka_bids instead."
        ),
        "parameters": {
            "operator_id": {"type": "string", "source": "session",
                            "description": "Operator identifier"},
            "market": {"type": "string", "source": "llm",
                       "description": "Market name (e.g. 'Kalyan', 'Milan Day', 'Rajdhani Day', 'Starline')"},
            "date": {"type": "string", "source": "llm",
                     "description": "Optional: date to fetch the result for (YYYY-MM-DD). Omit for today's/latest declared result."},
        },
        "default_path": "/operators/{operator_id}/matka-results",
        "method": "GET",
    },
    "get_operator_promotions": {
        "description": (
            "Get the operator's current promotions and bonus configuration: "
            "active promotions list, welcome/first-deposit bonus details and "
            "wagering requirements, cashback or losing bonus config, referral "
            "program details, VIP/loyalty tier definitions and per-tier benefits. "
            "NOTE: this does NOT return the player's personal referral code or "
            "link — call get_referral_code for those."
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
