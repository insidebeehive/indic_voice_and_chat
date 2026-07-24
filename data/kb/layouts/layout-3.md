# Layout 3 — Deltas from Global Baseline

Matka layout, same app/web split as layout 2 but different chrome and wallet UI. Live consumer: `mangal-matka-app` (**app builds only** — the layout-3 web variant is not used by any current app; mangal-matka's web build runs on layout 2). Config: `packages/layout-3/src/{app,web}/constant/app-configs/`.

## How it differs from layout 2 (same matka domain)

1. **Chrome**: `Header5Component` — hamburger + back on the left, **logo centered**, round phone/support button on the right (opens a support drawer; `layout.header.features.showSupport: true`). Page body sits on a colored header band with a rounded content sheet.
2. **Sidebar** (`SidebarComponent2`, fixed order): Profile (name + phone) → Game Rates → Terms & Conditions → Charts → Notifications → Change Password → Update PIN → language → Logout (confirm drawer) → "Invite Friends" (copies referral code; toast "Referral code copied!") → "Connect with us" socials. No Bonus entry in the sidebar (header bonus is also off: `layout.header.features.showBonus: false`) — the `/bonus` page exists but has no prominent nav entry.
3. **Bottom bar** (`Footer5Component`): items from the app's `constants/footer.ts` — in mangal-matka-app: casino crash-game shortcut · "All Bets" · Home · Wallet (live balance chip above the icon) · History · support. Hidden on `/sports` and `/casino` (`hiddenPaths`).
4. **Wallet UI is wallet-3**: a header with "Wallet Balance" and three tab buttons — "Deposit" / "Withdraw" / "Update Bank Details" — with an **on-screen numeric keypad** for amount entry ("Min: {min} | Max: {max}"), instead of layout 2's four-card hub with drawers. Deposit button label: "Add Funds"; withdraw: "Withdraw Funds"; success toast "Withdrawal request successful, Admin will take an action."
5. **Login onboarding** uses `OnboardUserComponent2` with an operator hero panel (`auth.onboard.backgroundImage/title/subTitle`, subtitle "Login to your account to continue") — same phone-first model as layout 2, different presentation.
6. Market cards are `MarketWidgetComponent3` with status strings "Betting is Open Now!" / "Betting is Open For Close!" / "Betting is Closed For Today!" (vs layout 2's "Running" wording) — both from `matka.displayConfig`.
7. Statements live at `/history` with tabs Wallet / Matka / Sports / Casino; the Sports tab is removed when `features.showSports` is off.
8. Starline/Jackpot are **off by default** in package configs (`matka.markets.*: false`); operators enable per config.
9. VoIP support is enabled for mangal-matka-app only (`voip.enableVOIP: true` with `voipBrandName`).

Everything else (MPIN, OTP flows, bid placement, charts, all-bids, game rates, bonus dialogs, self-exclusion, PG-only deposits, bank approval) matches the global docs and layout 2's matka model.

## Web variant

Same page set as layout 2's web variant (home, register, charts, about, how-to-play) — currently **unconsumed**; if a "layout-3 web" question comes up, answer per layout 2's web variant.
