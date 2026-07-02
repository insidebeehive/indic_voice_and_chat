# CRM API Contract — AI Support Chatbot Integration

**Audience:** Platform/CRM engineering team  
**Purpose:** The AI Support Chatbot calls these REST endpoints at runtime to answer player and operator queries. You implement these on your backend; we call them as tools during a support conversation.

---

## Overview

- All endpoints return **JSON**
- Authentication: **Bearer token** (sent as `Authorization: Bearer <token>`) — token is configured once during tenant onboarding
- All responses are free-form JSON; the AI reads and interprets them — no strict schema required, but the fields listed below are expected
- Failures (4xx/5xx) are handled gracefully — the AI will tell the customer it couldn't retrieve the information and offer to escalate

---

## Player Endpoints

These are called when a logged-in player starts a chat session. The chatbot automatically injects the player's `user_id` (from the session) and the `operator_id` (from your tenant config) — you do not need to accept these as user-provided inputs.

---

### 1. Player Wallet

```
GET /players/{user_id}/wallet?operator_id={operator_id}
```

**When called:** Player asks about balance, pending withdrawals, bonus balance.

**Expected response:**
```json
{
  "real_balance": 4250.75,
  "bonus_balance": 500.00,
  "total_available": 4750.75,
  "currency": "INR",
  "pending_withdrawal": {
    "amount": 1000.00,
    "status": "processing"
  }
}
```

`pending_withdrawal` may be `null` if no withdrawal is pending.

---

### 2. Player Transactions

```
GET /players/{user_id}/transactions?operator_id={operator_id}&type={type}&limit={limit}
```

**When called:** Player asks about recent transactions, deposit status, withdrawal history, game history.

**Query params set by AI:**
- `type` — `deposit` | `withdrawal` | `casino` | `sports` | `all` (default: `all`)
- `limit` — integer, default `10`

**Expected response:**
```json
{
  "transactions": [
    {
      "id": "txn_001",
      "type": "deposit",
      "amount": 2000.00,
      "status": "success",
      "timestamp": "2026-06-20T10:30:00Z",
      "method": "UPI"
    },
    {
      "id": "txn_002",
      "type": "casino",
      "amount": -350.00,
      "status": "settled",
      "timestamp": "2026-06-20T11:15:00Z",
      "game": "Teen Patti"
    }
  ],
  "total": 25
}
```

Negative `amount` = debit from player wallet.

---

### 3. Player Bets

```
GET /players/{user_id}/bets?operator_id={operator_id}&status={status}&limit={limit}
```

**When called:** Player asks about open bets, bet history, last bet result, winnings, cashout value.

**Query params set by AI:**
- `status` — `open` | `settled` | `all` (default: `all`)
- `limit` — integer, default `10`

**Expected response:**
```json
{
  "bets": [
    {
      "id": "bet_001",
      "sport": "Cricket",
      "match": "MI vs CSK",
      "selection": "MI Win",
      "stake": 500.00,
      "odds": 1.85,
      "status": "open",
      "cashout_value": 480.00,
      "placed_at": "2026-06-21T09:00:00Z"
    },
    {
      "id": "bet_002",
      "sport": "Cricket",
      "match": "IPL Final",
      "selection": "RCB Win",
      "stake": 300.00,
      "odds": 2.10,
      "returns": 630.00,
      "status": "won",
      "settled_at": "2026-06-20T14:00:00Z"
    }
  ],
  "pnl": {
    "this_week": 250.00,
    "this_month": -150.00
  }
}
```

---

### 4. Player Bonuses

```
GET /players/{user_id}/bonuses?operator_id={operator_id}
```

**When called:** Player asks about active bonuses, wagering requirements, bonus expiry, eligibility for new bonuses, referral earnings.

**Expected response:**
```json
{
  "active": [
    {
      "name": "Welcome Bonus",
      "amount": 500.00,
      "wagering_required": 2500.00,
      "wagering_completed": 800.00,
      "expires_at": "2026-07-05T00:00:00Z"
    }
  ],
  "welcome_bonus_claimed": true,
  "reload_eligible": true,
  "referrals": {
    "count": 3,
    "earnings": 300.00
  }
}
```

---

### 5. Player Profile

```
GET /players/{user_id}/profile?operator_id={operator_id}
```

**When called:** Player asks about VIP status/benefits, KYC status, registered contact details, saved bank/UPI, account age, login history.

**Expected response:**
```json
{
  "vip_tier": "Silver",
  "vip_benefits": ["Dedicated support", "Weekly cashback 5%", "Faster withdrawals"],
  "kyc_status": "verified",
  "kyc_documents": ["Aadhaar", "PAN"],
  "mobile": "+91-98765-43210",
  "email": "player@example.com",
  "bank_saved": {
    "bank": "HDFC",
    "account_last4": "7890",
    "upi": "player@upi"
  },
  "created_at": "2025-03-15T00:00:00Z",
  "last_login": "2026-06-21T08:45:00Z"
}
```

`kyc_status` should be one of: `verified` | `pending` | `rejected` | `not_started`  
`bank_saved` may be `null` if no bank account is saved.

---

### 6. Player Responsible Gaming

```
GET /players/{user_id}/responsible-gaming?operator_id={operator_id}
```

**When called:** Player asks about self-exclusion, deposit limits, betting limits.

**Expected response:**
```json
{
  "self_excluded": false,
  "self_excluded_until": null,
  "deposit_limit": {
    "daily": 10000,
    "weekly": 50000,
    "currency": "INR"
  },
  "bet_limit": null
}
```

`self_excluded_until` is an ISO 8601 timestamp if `self_excluded: true`, otherwise `null`.  
`deposit_limit` and `bet_limit` are `null` if no limits are set.

---

## Operator Endpoints

These are called for any question about how the platform works — payment methods, games, promotions, platform settings. Only `operator_id` is injected (no `user_id`).

---

### 7. Payment Configuration

```
GET /operators/{operator_id}/payment-config
```

**When called:** Questions about deposit/withdrawal methods, min/max amounts, supported banks, UPI support, processing time.

**Expected response:**
```json
{
  "deposit_methods": ["UPI", "Net Banking", "Debit Card", "Paytm"],
  "withdrawal_channels": ["Bank Transfer", "UPI"],
  "deposit_limits": { "min": 100, "max": 100000, "currency": "INR" },
  "withdrawal_limits": { "min": 500, "max": 50000, "currency": "INR" },
  "supported_banks": ["HDFC", "ICICI", "SBI", "Axis"],
  "blocked_banks": ["PMJDY accounts"],
  "upi_supported": true,
  "withdrawal_processing_time": "24-48 hours (business days)"
}
```

---

### 8. Games Configuration

```
GET /operators/{operator_id}/games-config
```

**When called:** Questions about available games, sports, live casino, Matka, virtual sports, in-play betting.

**Expected response:**
```json
{
  "casino_providers": ["Evolution Gaming", "Ezugi", "Pragmatic Play"],
  "sports": ["Cricket", "Football", "Tennis", "Kabaddi"],
  "leagues_covered": ["IPL", "T20 World Cup", "EPL"],
  "live_casino": true,
  "matka_available": true,
  "virtual_sports": true,
  "lottery_games": ["Kerala Lottery", "Bhutan Lottery"],
  "in_play_betting": true,
  "cashout_available": true
}
```

---

### 9. Promotions & VIP

```
GET /operators/{operator_id}/promotions
```

**When called:** Questions about welcome bonus, cashback, active promotions, referral program, VIP tiers and benefits.

**Expected response:**
```json
{
  "active_promotions": [
    {
      "name": "Welcome Bonus",
      "type": "first_deposit",
      "offer": "100% match up to ₹10,000",
      "wagering_requirement": "5x",
      "min_deposit": 500,
      "valid_until": "2026-12-31"
    },
    {
      "name": "Refer & Earn",
      "type": "referral",
      "offer": "₹100 bonus per verified referral",
      "max_referrals": 50
    }
  ],
  "vip_tiers": [
    { "tier": "Bronze",   "min_deposits": 0,      "benefits": ["Standard support"] },
    { "tier": "Silver",   "min_deposits": 5000,   "benefits": ["5% weekly cashback", "Priority support"] },
    { "tier": "Gold",     "min_deposits": 25000,  "benefits": ["10% weekly cashback", "Dedicated manager"] },
    { "tier": "Platinum", "min_deposits": 100000, "benefits": ["15% cashback", "Instant withdrawals"] }
  ]
}
```

---

### 10. Platform Configuration

```
GET /operators/{operator_id}/platform-config
```

**When called:** Questions about currencies, languages, support contacts, hours, KYC requirements, geo restrictions, mobile app, brand info.

**Expected response:**
```json
{
  "currencies": ["INR"],
  "languages": ["Hindi", "English", "Telugu", "Tamil", "Marathi"],
  "timezone": "Asia/Kolkata",
  "minimum_age": 18,
  "support_contacts": {
    "phone": "+91-800-123-4567",
    "email": "support@yourplatform.com",
    "whatsapp": "+91-900-123-4567",
    "live_chat": true
  },
  "support_hours": "24x7",
  "mobile_app": { "android": true, "ios": false },
  "kyc_documents_required": [
    "Government photo ID (Aadhaar / PAN / Passport / Voter ID)",
    "Proof of address"
  ],
  "geo_restrictions": ["Not available in Telangana, Andhra Pradesh for certain games"],
  "brand": "Your Platform Name",
  "operator_profile": "Brief description of the platform."
}
```

---

## Escalation

The following query types are **not handled by API lookups** — the chatbot escalates to a human agent via the built-in escalation flow:

| Customer query | Escalation reason |
|---|---|
| "My deposit was deducted but not credited" | Manual transaction investigation |
| "My withdrawal was rejected" | Needs review of rejection reason |
| "Someone accessed my account" | Security incident — account freeze required |
| "I want to close my account" | Human confirmation + balance settlement |
| "I want to dispute a bet settlement" | Manual review against provider data |
| "I was charged twice" | Payment reconciliation required |
| "I want to remove my self-exclusion early" | Responsible gaming policy review |

When the bot decides to escalate, it posts an `escalation_requested` event to your configured `events_webhook_url`:

```json
{
  "event": "escalation_requested",
  "session_id": "cs_a1b2c3d4",
  "reason": "Customer requested human support",
  "summary": "Customer is asking about a 3-day withdrawal delay.",
  "customer": { "name": "Rahul", "id": "player-42" },
  "claim_url": "/api/v1/chat/sessions/cs_a1b2c3d4/claim",
  "agent_ws_url": "/api/v1/chat/sessions/cs_a1b2c3d4/agent-ws",
  "bo_available": true,
  "event_id": "evt_9f3b1a2c"
}
```

`bo_available` reflects our knowledge of your support-hours config. Your system makes the final availability decision.

After receiving this webhook you **must** call exactly one of the two endpoints below. If neither is called, the customer is left waiting indefinitely.

### Claim the session (agent available)

```
POST /api/v1/chat/sessions/{session_id}/claim
Authorization: Bearer <tenant-token>
Content-Type: application/json

{
  "agent_id": "agent_007",
  "agent_name": "Priya"
}
```

**Response:**
```json
{ "status": "claimed", "agent_id": "agent_007" }
```

The customer is immediately notified that an agent has joined. The agent then connects to `agent_ws_url` to exchange messages.

Returns `409` if already claimed, `400` if the session is not in `awaiting_human` state.

### Decline the session (no agents available)

If you cannot assign an agent (outside business hours, queue full, etc.), POST to the decline endpoint so the bot resumes the conversation:

```
POST /api/v1/chat/sessions/{session_id}/decline
Authorization: Bearer <tenant-token>
```

**Response:**
```json
{ "status": "declined" }
```

The customer receives an apology message and the AI bot resumes the conversation. Returns `400` if the session is not in `awaiting_human` state.

---

## Onboarding

Once your endpoints are live, register them with our platform in one call:

```http
POST /api/v1/chat/tools/from-catalog
Authorization: Bearer <your-tenant-token>
Content-Type: application/json

{
  "crm_base_url": "https://api.yourplatform.com",
  "auth_type": "bearer",
  "auth_token": "<crm-api-token>",
  "tools": [
    "get_player_wallet",
    "get_player_transactions",
    "get_player_bets",
    "get_player_bonuses",
    "get_player_profile",
    "get_player_responsible_gaming",
    "get_operator_payment_config",
    "get_operator_games_config",
    "get_operator_promotions",
    "get_operator_platform_config"
  ]
}
```

Omit the `tools` array to register all 10 tools at once. This call is idempotent — safe to re-run when your base URL or token changes.
