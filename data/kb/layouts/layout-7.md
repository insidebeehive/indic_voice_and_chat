# Layout 7 — Deltas from Global Baseline

Sports-first responsive layout. Consumers: `bigbull-v2` (+ `layout7-app` template; configs intentionally identical for now). Config: `packages/layout-7/src/constant/app-configs/`.

## Navigation

1. **Desktop**: persistent left **sports sidebar** ("All Sports" + one link per sport → `/sports?sport={name}`). Header (`Header9Component`): logo · search pill · nav "Sportsbook" / "Casino" (→ `/live-casino`) / "Promotion" · support pill · balance chip (cash + bonus) · green "Deposit" · language · avatar menu. Guests see **both "Login" and "Register"** buttons in the header (unique among layouts).
2. **Mobile**: the hamburger opens the **sports sidebar** (a sports list — not a site menu); site navigation happens via the bottom bar.
3. **Bottom bar** (`MobileFooter11`): HOME · CASINO · center **WhatsApp support FAB in a curved notch** · SPORTSBOOK · PROFILE (opens a profile drawer: Profile, Wallet, Statements, Promotion, Bank Details with pending badge; guests: Login).
4. **Floating action button** (right edge, `layout.floatingActionButton`): quick menu with Profile / Bonus / Account Statement / balances / Support / Logout (guests: Login).
5. Footer tagline: "Your trusted platform for sports exchange, live casino, and premium gaming entertainment. Play responsibly." with RNG/18+/Responsible Gaming badges. Footer hidden on `/sports` and `/casino/play`.

## Home page

Sports-first dashboard: live-odds widget with sport tabs and event cards ("team1 vs team2", LIVE badge, 6-cell back/lay grid; empty "No events available"; cells deep-link into `/sports`), followed by casino content (menu chips and emoji-labelled rows "🕹️ Recently Played", "🔥 Trending Games", "🏆 Top Games", "❤️ Favourite Games").

## Auth deltas

1. Login header: "Login to your account" / "Welcome {operatorName}" (`LoginComponent4`).
2. Register: two-column single-page form (`RegisterComponent1`, "Register your account"); success screen "Registration Successful!" / "You will be redirected to the home page now." (3s). Note: the config declares `auth.components.registerComponent: RegisterComponent5`, but the register page does not read that knob — it's currently a no-op.

## Wallet deltas

Base wallet form (wallet-1, same as layout 1): quick-amount grid, "Amount (min: …)" field, a "History" button opening a slide-over of the last five transactions, side-by-side green "Deposit" / red "Withdraw" buttons. Bank details at `/wallet/addbank`.

## Other

1. **Legal**: ten dedicated static routes (`/kyc`, `/privacy`, `/termsofservice`, …) linked from the footer — no `/legal-policy` hub. `/responsiblegaming` includes the self-exclusion entry card; `/selfexclusion` shows the form plus policy text.
2. Bonuses at `/promotion` (`BonusComponent4`). The floating button also links to `/bonus`, which has **no route in the template app** — it 404s unless an operator app adds it [INFERRED].
3. Casino browsing at `/live-casino` with Provider + Games filter dropdowns; vendor names shown on cards.
4. Statements: `StatementsComponent1` with a Matka drill-down action wired (Matka tab appears only if backend data includes it).
5. No matka pages, no profile hub page (drawer/avatar menu/FAB instead).
