# Layout 8 — Deltas from Global Baseline

Sports-centric, mobile-first layout **under active build-out** — no live operators yet (template app `layout8-app` only). Several sections are placeholder pages. Config: `packages/layout-8/src/constant/app-configs/layout8-app.ts`.

## Current state (important for support)

The following routes render "coming soon" placeholders: `/casino` and `/casino/:menuId` (browsing), `/live-casino`, `/sports`, `/promotion`, and the wallet result pages (`/wallet/{success,failure,complete}`). Casino **game play** works (`/casino/play/...`), and home-page casino shelves are live. The bottom-nav "My Bets" item points at `/my-bets`, which has no route yet — it lands on the 404 page [INFERRED].

## Navigation

1. **Header** (`Header10Component`): back arrow · logo · (logged in) "Real" balance pill (tap to refresh) + wallet icon. No auth buttons in the header.
2. **Bottom nav** (`MobileFooter12`, always visible): Home (`/home`) · My Bets · My Wallet · Profile. Hidden on `/login`, `/register`, `/register-success`, `/casino/play` (`footer.hideOnRoutes`).
3. **Menu sheet** (`MobileMenu2`): Casino · Sports · Promotions · collapsible "Usage & Terms" (ten legal links) · language · "Live Chat" · socials. Logout lives on the Profile page.
4. **Home** (`/home`): "Hi {name}, Welcome Back", search, avatar; two tabs "Bet Sports" / "Bet Casino"; casino shelves (Recently Played / Trending / Top / Favourite).
5. Landing `/`: splash "Unleash your Betting Potential" with "Get Started" (auto-redirects after ~3s; `landingPage: LandingPageComponent3`).

## Screens

1. Auth: "Sign in to your account" (`LoginComponent6`, strategy stepper) and "Sign up to your account" (`RegisterComponent7`).
2. Wallet: single-form `WalletComponent4` (as layout 5); bank managed at `/accountsettings?bankOnly=true`.
3. Profile page (`UserProfileComponent4`): Profile Edit → `/accountsettings` (stacked Profile/Bank/Address forms), Account Statement, Bonus, Security → `/updatepassword` ("Set New Password"), Legal Policy, Support, Log Out.
4. Bonus page title "Bonus and rewards" (`BonusComponent3`, tabs Casino Bonus / Bonus History).
5. Legal: single `/legal-policy` hub with dropdown (as layouts 5/6).

## Config notes

Single config key. `features.showSportsFeedWidget: true` is declared but no component reads it (no-op). `casino.menu.show: true` is set although the casino listing page is stubbed.
