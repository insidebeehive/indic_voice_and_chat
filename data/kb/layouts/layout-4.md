# Layout 4 — Deltas from Global Baseline

**Teer (archery lottery) variant of layout 3.** Live consumers: `meghateer-app` (app) and `meghateer-web` (web). Config: `packages/layout-4/src/{app,web}/constant/app-configs/`.

## Relationship to layout 3

Layout 4 re-exports 16 pages directly from layout 3 (login, wallet + bank + history, bonus, notifications, sports, update password/MPIN, referrals, all-bids, game rates, terms, 404). **Auth, wallet, deposit/withdrawal, bonus, and profile answers are identical to layout 3** — use `layouts/layout-3.md` and the global docs for those.

## Teer game domain (replaces matka home)

1. **Home dashboard**: one card per Teer game — name, status line, two result columns "FR ({time})" and "SR ({time})" (result shown as "FR-SR"), arrows-hit stats "{hit} / {distributed} ({players})", buttons "Official Result", club-chart icon, and "Play Now →" (green when open, red when both rounds closed).
2. **Game-type chooser** (`/market/:marketId`): tiles "Play Gutti", "Play Housing & Ending", "FC", "Line", "Previous Result".
3. **Bid page** (`/market/:marketId/:betType`): "Playing {type} In {gameName}" with a countdown timer, FR/SR session tabs (or Single FC / Multi FC), and bet-type-specific panels (Housing & Ending uses "Left Digit"/"Right Digit"). If the round closed: "You are late" / "{session} is now closed". Invalid game-type URL: "Invalid Request" / "The requested game type is not available. Please check the URL and try again."
4. **Results** (`/results/:gameName`): table with Date / FR / SR columns; empty "There are no results available!".
5. **Charts**: Teer FR/SR history charts plus a club-chart image preview page ("Club Chart Preview" / "Click to view full chart").
6. Matka-style bid pages (`/matkagames/:bettype`) remain available alongside Teer.

## Other deltas vs layout 3

1. **Statements**: the game tab is "Teer" (type `UDTATEER`; empty state "No Teer transactions found for selected date range!") instead of "Matka", and the Sports tab is **not** filtered by `features.showSports`.
2. **Bottom nav is never hidden** — layout 4's page layout does not implement route hiding (layout 3 hides the bar on `/sports`/`/casino`).
3. Profile page drops the member-since subtitle.
4. No VoIP config in any layout-4 config.

## Web variant (meghateer-web)

Public Teer brochure site: home (adverts, "Game Rates" pricing, "Available Games" with app-download prompts), How to Play ("Players must predict the last two digits of the total number of arrows that hit the target.", FR/SR explanation), About Us, public charts, `/register` (+ success page prompting app download), disabled-registration page, agent-link landing. No login/wallet/betting.
