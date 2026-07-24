# Layout 5 — Deltas from Global Baseline

Mobile-first casino app rendered as a fixed **430px centered column** even on desktop. Consumers: `spade91-v2` (+ `layout5-app` template; configs currently identical). Config: `packages/layout-5/src/constant/app-configs/`.

## Navigation

1. **Header** (`Header6Component`): centered logo, contextual back button, and for logged-in users an **animated rotating balance pill** (cycles balance categories) plus a wallet icon → `/wallet`. **No nav links and no hamburger** — the `/profile` page acts as the menu.
2. **Bottom tab bar** (`MobileFooter6`, icon-only, no labels): Home (operator's default casino URL) · Sports · Wallet · Profile when logged in; guests get Register + Login instead (`showSimpleFooterForGuests`, Register gated by `features.registerAvailable`). Hidden on `/casino/play` and `/sports` (`hideOnRoutes`).
3. Casino home is `/live-casino`.

## Screens

1. **Profile hub** at `/profile`: avatar + "Edit profile" → `/accountsettings`; cards Bonus / Account Statement; Account section (Profile Edit with bank-pending badge, Change Password → `/updatepassword`, Search); support chat; language; Legal Policy; socials; Log Out.
2. **Account settings** stacks Update Profile / Address / Bank forms plus the self-exclusion card; `?bankOnly=true` shows only the bank form (this is where the wallet's "Add/Manage" bank link goes — there is no `/wallet/addbank`).
3. **Wallet** (`WalletComponent4`): single "Amount*" field with Deposit and Withdraw buttons (no tabs/quick-chips).
4. **Legal**: single `/legal-policy` hub page with a dropdown of the ten policy topics; `/selfexclusion` separate.
5. **Bonus**: both `/bonus` (tabs "Casino Bonus" / "Bonus History") and `/promotion` (tabs "Advertisement" / "Bonus").
6. Auth: `LoginComponent4` ("Sign in to your account" / "Welcome back to {operatorName}") + single-page `RegisterComponent5` ("Create an Account").
7. Casino game grid: 3 per row with game names hidden on cards (`casino.gameDisplay.hideGameNameOnCard: true`).

## Not present

Matka, desktop chrome of any kind, hamburger menu, `/wallet/addbank`.

## AppConfig knobs

`layout.header.styling` (gradient links/buttons, outlined balance widget, centered logo), `features.showSportsFeedWidget` ("Live Games" odds widget on casino home), `search.variant: 'gradient'`, `landingPage: LandingPageComponent1` (splash auto-redirect). Both current configs are identical — no per-operator divergence yet.
