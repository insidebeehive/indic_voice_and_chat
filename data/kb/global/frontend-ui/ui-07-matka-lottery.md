# Matka / Lottery — Shared UI Behavior

## Availability by layout

Matka UI exists **only on the matka layouts (2, 3) and the Teer variant (4)**. Casino/sports layouts (1, 5–9) have no matka screens (at most a Matka tab in statements when backend data exists). Resolve the operator's layout first (`operator-to-layout.md`).

## Shared matka model (layouts 2 and 3)

The two matka layouts differ in visual chrome (see their layout docs) but share the same flow:

### Home market list

1. The home page lists one card per market: market name, latest result digits, a status line (operator/layout-configurable via `matka.displayConfig` — e.g. "Betting is Running Now" / "Betting is Open Now!", a "running for Close" variant, and a red closed-for-today variant), Open/Close result times, a chart shortcut, and a Play button (disabled/red when closed).
2. Optional **Starline** and **Jackpot** market sections appear when the operator enables them (`matka.markets.showStarline` / `showJackpot`).
3. Empty state: "No games available".

### Placing a bid

1. Tap Play on an open market → a **bet-type grid** (Single Digit, Jodi Digit, Single/Double/Triple Pana, bulk variants, SP/DP Motor, Group Jodi, SP/DP/TP, Odd Even, Red Bracket, Digit Based Jodi, Choice Pana, Panel Group, Two Digit Panel, Half Sangam, Full Sangam). Starline/Jackpot markets show reduced sets. Jodi-family types are unavailable when only the Close session remains.
2. Pick a bet type → bid entry page: read-only date, "Select Session" dropdown (Open/Close; Starline/Jackpot fixed), and bet-type-specific digit/points inputs (bulk types generate multiple bids).
3. A sticky footer totals the bids ("Total Bids" / "Total Amount") with a "Place Bet" button.
4. A confirmation drawer summarizes Digit / Points / Game Type rows and the wallet balance, and warns that bids cannot be cancelled once placed. Confirm with "Submit Bet"/place action.
5. Validation toasts: "Minimum bid amount is {currency}{amount}", "Maximum bid amount is {currency}{amount}", "Betting is Closed for today!", plus per-type input validation.

**Bids are not cancellable** — there is no cancel-bid UI anywhere.

### Results & charts

1. Each market card shows the latest result string.
2. A Charts page lists per-market "Panna Charts" and "Jodi Charts" (plus Starline/Jackpot charts when enabled) → historical chart tables. Matka **web** variants expose charts publicly without login.
3. Empty state: "No chart records are available".

### Bid history & rates

1. "All Bids" page: paginated bid cards with a date-range filter drawer. Empty: "No Bid History Found" / "You haven't placed any bids yet.".
2. "Game Rates" page: "Game Win Rates" — bet type → payout rate list (operator values).

## Teer (layout 4 only)

Layout 4 replaces the matka domain with **Teer** (archery lottery): FR/SR rounds, bet types Gutti / Housing & Ending / FC / Line, official results tables, and club charts — see `layouts/layout-4.md`. Matka-style bid pages remain available alongside.

## App vs web variants (layouts 2, 3, 4)

Each matka operator typically has an **app** build (login-gated play) and a **web** build (public brochure site: register, charts, how-to-play, about; no login/wallet/betting). If a user is on the web site, actual play happens in the downloadable app — the web register success page says so: "You have successfully registered, Please download app now and start playing!".
