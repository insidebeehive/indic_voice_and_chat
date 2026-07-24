# Layout 2 — Deltas from Global Baseline

Matka layout with **two variants**: `src/app/` (login-gated PWA where all play happens) and `src/web/` (public brochure site). Config: `packages/layout-2/src/{app,web}/constant/app-configs/`.

## App variant

### Navigation

1. **Header** (`Header2Component`): hamburger (sidebar) or back arrow · logo/page title · animated balance (tap to refresh; labeled "Real" or "Bonus") · wallet icon → `/wallet` · optional support headset (`layout.header.features.showSupport`, e.g. shree-matka-app).
2. When the operator enables sports (`features.showSports`), a **Matka | Sports tab switcher** appears under the header on `/` and `/sports`.
3. **Sidebar** (fixed order): User Profile → Bonus → Game Rates → Terms & Conditions → Charts → Notifications → Change Password → Update PIN → Videos (toast "coming soon") → language → Logout.
4. **Bottom bar** (`Footer2Component`) items come from the consuming app's `footerConfig` — typical order: History · All Bids · Home · Wallet · Support; may include a round casino-game shortcut. Hidden on paths in `hiddenPaths` (e.g. `/casino`, `/sports`).

### Auth deltas

1. **Phone-first onboarding** at `/login`: "Enter Number" → "Continue" → server routes to login (existing) or inline registration (new number). OTP-or-password login with switch links.
2. **MPIN lock**: after login, an "MPIN Verification" dialog appears on app restart and after ~2 minutes of inactivity (not on game-play pages; not for demo users). "Forgot PIN?" runs an OTP-verified MPIN reset. Update at `/updatempin` (numeric keypad).
3. Password changes at `/updatepassword` (OTP-verified).

### Wallet deltas

1. `/wallet` is a hub of **four cards**: "Add Fund" (drawer) · "Withdraw Fund" (drawer) · "Deposit & Withdraw History" (page) · "Add Bank Details" (`/wallet/addbank`).
2. Withdrawal drawer always shows "👉 Withdrawal limit is {currency}{min} to {currency}{max}" and "👉 Withdrawal Request timing is {start} to {end}".

### Matka specifics

1. Home shortcut pills: **Starline** → `/starline`, **Games** (collapsible casino tiles), **Jackpot** → `/jackpot` — gated by `matka.markets.showStarline`/`showJackpot` (enabled for matka-app/bharat-matka-app).
2. Market status strings from `matka.displayConfig`: "Betting is Running Now" / "Betting is Running For Close" / "Betting is Closed For Today".
3. Market card widget varies per brand: `matka.components.marketWidget` (Widget1 vs Widget2).
4. Notification settings page offers Starline/Jackpot notification toggles when those markets are on.

### Other app-variant notes

- Profile page carries the referral program (referral code card, "Total Income", referral lists) when `auth.referral` is enabled.
- Statements page is `/history` with tabs Wallet / Matka / Sports / Games (Sports removed when sports is off).
- Casino: no lobby — home "Games" panel and bottom-bar shortcut launch games directly.

## Web variant

Public brochure site only — **no login, wallet, betting, bonus, or profile**:

1. Header (`Header4Component`) menu: Register · How to Play · Charts · About Us (+ optional Telegram/WhatsApp buttons, language).
2. `/register` is a full public form → OTP → success page: "You have successfully registered, Please download app now and start playing!" plus app-download CTAs.
3. Charts are public (no login).
4. Footer (`Footer4Component`): 18+ notice, operator address/licence text (`footerWebConfig`), socials.

## AppConfig knobs that vary per operator

| Field | Effect | Examples |
|---|---|---|
| `features.showSports` | Sports tab switcher + `/sports` + Sports statements tab | matka-app |
| `layout.header.features.showBonus: false` | Hides bonus in header | rama567-app |
| `layout.header.features.showSupport: true` | Header support button | shree-matka-app |
| `matka.markets.showStarline/showJackpot` | Starline/Jackpot sections | matka-app, bharat-matka-app |
| `layout.footer.mobileNav.items/hiddenPaths` | Bottom-bar contents | per consuming app |
| `auth.referral.*` | Referral field + profile widgets | matka-app |
| `voip.enableVOIP` | VoIP support widget | per config |
| `appDownloadConfig` | App-download banner/CTAs | per app |
