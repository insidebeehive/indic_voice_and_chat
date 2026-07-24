# Layout 6 — Deltas from Global Baseline

Fully responsive casino-first layout with the **highest per-operator variance** — headers, footers, casino menu components, and the register form all swap per config. Consumers: `aceplay7`, `bettingraja247`, `pridegame99` (+ `layout6-app` template). Config: `packages/layout-6/src/constant/app-configs/`.

## Navigation

1. `/` redirects into the operator's default casino menu; `/home` is a richer casino landing (hero carousel, provider grid with an "Explore All" tile, game rows).
2. **Desktop header** (`Header7Component` default): logo · nav pills "Casino" / "Live Casino" / "Promotions" / "Exchange" (→ `/sports`) · search · balance capsule (Cash + Bonus) · green "Deposit" CTA · language · avatar menu (Profile, Wallet, Statements, Bank Details with pending badge, Log Out). Guests: single "Login" (or "Login/Demo") button.
3. **Mobile**: hamburger opens `MobileMenu2` (Casino / Sports / Promotions, collapsible "Usage & Terms" with all ten legal links, language, "Live Chat", socials); avatar opens a bottom-sheet profile drawer (Cash/Bonus/Exposure cards + menu).
4. **Bottom bar** (`MobileFooter9`, template app): Sports · Casino · Support · **In Play** (`/sports/in-play`). Operator apps instead use a full `Footer10Component` (no tab bar).
5. Operator apps swap in `Header8Component` — a premium black/red/gold header skin (`layout.header.component`).

## Auth deltas

1. **Register is a 4-step wizard** by default (`RegisterComponent6`): steps "Personal" → "Contact" → "Security" → "Confirm" (review + terms) → Send OTP. All three live operators override to the single-page form via `auth.components.registerComponent: RegisterComponent5`.
2. "Login with OTP" is offered **only for the phone strategy** (hidden for Username/Email logins).
3. bettingraja247: `auth.defaultAuthStrategy: 'USERNAME'` (login opens on Username) and `redirectAfterPasswordReset: true`.

## Wallet deltas

`WalletComponent5`: balance rows ("Real Balance", bonus, "Exposure" when the operator runs exchange sports) · a **Deposit | Withdraw segmented toggle** · quick-amount chips · amount field with "Min: {min} · Max: {max}" helper · CTA "Deposit Now" / "Withdraw Now". Bank details on a dedicated `/wallet/addbank` page.

## Casino deltas

Multiple interchangeable casino IAs are shipped and selected per operator app: category-wise pages with drill-down tables ("{category} ({count})", "Search games..."), a provider-first page with an "All Category" filter modal ("Pick Your Category" / "Apply"), a "Choose a Provider" landing, and a combined casino+sports landing. Casino menu component varies per operator (`casino.components.menuComponent`: `CasinoMenuComponent4` for aceplay7/pridegame99, `CasinoMenuComponent2` for bettingraja247).

## Other

1. Bonuses live at `/promotion` only (`BonusComponent4` — filterable grid, "Showing {n} bonus(es)"); no `/bonus` route.
2. No profile hub page — profile actions are in the avatar menu / drawer. Account settings (`EditProfilePageComponent1`) has a balance sidebar, profile + "Security" password cards, and the self-exclusion card.
3. Statements use the desktop-grade `StatementsComponent1` (type select, 45-day range, drill-down dialogs).
4. Legal: `/legal-policy` hub (same as layout 5) plus the mobile menu's "Usage & Terms".
5. Registration-disabled page copy differs here: "Registration Disabled" / "New account registration is currently unavailable." / "Please contact our support team for assistance."
6. No matka.
