# Layout 1 — Deltas from Global Baseline

Casino-first desktop+mobile layout; the largest operator base (see `operator-to-layout.md`). Config: `packages/layout-1/src/constant/app-configs/`.

## Navigation

1. `/` redirects straight into the default casino route — there is no separate home page (some operators show a landing screen first, below).
2. **Desktop header** (`Header1Component`): logo · game search · text-resize button · a context toggle that reads "Sports" on casino pages and "Live Casino" on `/sports` · support button · (logged in) "Bonus", "Wallet", balance widget, language, avatar menu · theme switcher. Logged out: "Login" (labelled "Login/Demo" when demo is on) + "Register".
3. **Avatar menu**: Account Settings (with bank-change badge) · Bonus · Deposit · Withdraw (both go to `/wallet`) · Account Statement · Logout.
4. **Mobile bottom bar** (`MobileFooter1`): Casino · Sports · center Search (opens the mobile menu sheet) · Wallet (guests: "Log In") · Support. With `layout.footer.mobile.showSimpleFooterForGuests`, guests instead get Register / Log In / Support. Hidden on `/sports`.
5. **Mobile menu** (`MobileMenu1` sheet): welcome line, search, wallet info + Deposit/Withdraw, Bonus, Account Settings, Account Statement, casino menu list, language, Logout.
6. Desktop footer link columns are effectively absent — no layout-1 operator config sets `layout.footer.navigation`, so only license/copyright/mobile content renders.

## Screens & flows unique to this layout

1. **Landing page**: default `LandingPageComponent1` is a splash that auto-redirects to casino after ~2s; operators with `landingPage: LandingPageComponent2` (kbcexchange, cbtfstake, khelomama, vipbets) get a login-card landing instead.
2. **Rewards** (`/rewards/claim`): redeem reward points into a bonus ("Redeem Now", minimum-points note, irreversible-redemption confirm). Errors: "Insufficient points", "You have not updated your address".
3. **Sports list** (`/sportslist`): "Select your favorite sport" — only for operators with `enableSportsList` (registered e.g. in kbcexchange).
4. Ten dedicated static/legal routes (`/termsofservice`, `/privacy`, `/kyc`, …).
5. Auth: full `/login` + `/register` pages (`LoginComponent1`/`RegisterComponent1`). Wallet: single combined form (wallet-1) with quick-amount buttons and a "History" slide-over (last 5 transactions). Bank details are edited in **Account Settings** (not a wallet subpage).

## Not present

Matka UI (no operator config sets `matka`), KYC upload, forgot-password page, native betslip (sports is the embedded sports app).

## AppConfig knobs that vary per operator

| Field | Effect | Examples |
|---|---|---|
| `features.registerAvailable: false` | Hides Register CTAs | mgm91vip, playbooth |
| `layout.header.component: Header3Component` | Alternate header skin | karavip, shakunivip |
| `layout.footer.mobile.component: MobileFooter8` | Alternate bottom bar | kingsplay |
| `layout.floatingActionButton` | Floating quick-action button | karavip, shakunivip |
| `landingPage: LandingPageComponent2` | Login-card landing | kbcexchange, cbtfstake, khelomama, vipbets |
| `auth.defaultAuthStrategy` | Preselected login tab | mgmfun |
| `casino.menu.variant/size/hideCategory/drawerItems` | Casino menu styling & behavior | various |
| `search` | Search button styling | ~22 configs |
