# Layout 9 — Deltas from Global Baseline

Casino-first layout with a persistent **desktop sidebar**. Consumers: `goaroyals` (+ `layout9-app` template; configs currently identical). Config: `packages/layout-9/src/constant/app-configs/`.

## Navigation

1. **Desktop**: collapsible left sidebar (Bonuses → `/promotion`, Account Statement, All Providers, live casino menus, Support, Terms, theme toggle). Header (`Header11Component`): logo · search pill ("Search games, providers, sports…") · support button · guests "Log In" (+"/Demo") and "Register" · logged-in avatar pill → `/accountsettings` and a "Deposit" button → `/wallet`.
2. **Mobile bottom nav** (`MobileFooter13`): Menu (opens the sidebar as a sheet) · Providers · center **Wallet FAB** · Support · Search. Hidden on `/sports` and `/casino/play`.
3. **Floating action button** (right): popover with Real/Bonus balances, Profile, Bonus, Account Statement, language, Support, Logout.
4. **Desktop footer** (`Footer13Component`): four legal columns (all ten policy routes), "Get the app" card, 18+ / "Play responsibly", Curaçao licence notice.
5. Home: hero carousel → backend-curated shortcut tiles → sports pills → per-menu casino rows → providers grid.

## Auth deltas

1. `/login` and `/register` render a **combined tabbed auth card** ("Register" | "Log In"); this operator family uses the "Royal Vault" gold-pill variant (`auth.components.authTabs: AuthTabsComponent2`, register `RegisterComponent5`, strategy toggle "By Phone" / "By Email" pills).
2. Demo mode: "Try Demo Mode" on the login card; amber "You're in Demo Mode." banner and guard modal apply while in demo.

## Wallet deltas

**Tabbed wallet** (`WalletComponent6`, deep-linkable `?tab=`):
1. **Balance**: total balance, cash + bonus pills, Deposit/Withdraw buttons, bank status, recent transactions.
2. **Deposit**: amount with "Min/Max" hint, "Quick add" chips, "Add Funds" → "You'll be redirected to your payment provider to complete this deposit."
3. **Withdraw**: "Withdrawable" balance (with "Locked in active {bonus} wagering" note), "Withdraw Timing" window, bank status with "Manage"/"Add" → `/accountsettings?tab=bank`, OTP confirm.

## Other

1. **Account settings** (`/accountsettings`): four deep-linkable tabs — Profile / Password / Bank / Address (`?tab=…`).
2. **Casino browsing** is the primary vertical: `/casino/:menuId/:categoryId` with Provider and Games filters, tag pages (`/casino/tag/:tag`), provider pages (`/providers`, `/providers/:providerId` — "This provider has no games yet.").
3. Bonuses at `/promotion` ("Bonuses & Promotions": Active with Cashout/Remove; Available with "Claim {amount}"/"Participate"/"NA" when locked).
4. Statements page titled "History" (`StatementsComponent5`): underline tabs (WALLET/CASINO/SPORTS/MATKA per operator data), "Pick dates" popover + quick day pills.
5. **Legal**: ten dedicated static routes (like layouts 1/7), reachable from footer, sidebar, and mobile menu.
6. Sports via `/sports` iframe. No matka pages.
7. Config references `/wallet/addbank` and `/bonus` in menu items but the template app has no such routes [INFERRED pending]. `search.showRecommendedGames: true` currently has no effect (shelf commented out).
